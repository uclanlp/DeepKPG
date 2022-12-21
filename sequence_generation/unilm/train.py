from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import shutil
import json
import random

import numpy as np
import torch
from torch.utils.data import (DataLoader, SequentialSampler)
from torch.utils.data.distributed import DistributedSampler

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

import tqdm

from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
    RobertaConfig,
    BertConfig,
    BertTokenizer,
    RobertaTokenizer,
    XLMRobertaConfig,
    XLMRobertaTokenizer,
    AutoConfig,
    AutoTokenizer
)
from sequence_generation.unilm.source.modeling import BertForSequenceToSequence
from sequence_generation.unilm.source.configuration_unilm import UnilmConfig
from sequence_generation.unilm.source.tokenization_unilm import UnilmTokenizer
from sequence_generation.unilm.source.configuration_minilm import MinilmConfig
from sequence_generation.unilm.source.tokenization_minilm import MinilmTokenizer
from sequence_generation.unilm.source.configuration_xbert import xBertConfig
from sequence_generation.unilm.source.tokenization_xbert import xBertTokenizer, xBertTokenizerRoberta

from sequence_generation.unilm.source import utils
from sequence_generation.unilm.source.config import BertForSeq2SeqConfig

from copy import deepcopy

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'bert': (BertConfig, BertTokenizer),
    'minilm': (MinilmConfig, MinilmTokenizer),
    'minilm2-bert-base': (BertConfig, BertTokenizer),
    'minilm2-bert-large': (BertConfig, BertTokenizer),
    'minilm2-roberta': (RobertaConfig, RobertaTokenizer),
    'roberta': (RobertaConfig, RobertaTokenizer),
    'xlm-roberta': (XLMRobertaConfig, XLMRobertaTokenizer),
    'unilm': (UnilmConfig, UnilmTokenizer),
    'xbert': (xBertConfig, xBertTokenizer),
}

TOKENIZER_NAMES = {
    'minilm2-bert-base': 'bert-base-uncased',
    'minilm2-bert-large': 'bert-large-uncased',
    'minilm2-roberta': 'roberta-large',
}


def prepare_for_training(args, model, amp):
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.num_training_steps, last_epoch=-1)

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
            os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    if amp:
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
                                          #loss_scale=1.0)  # TODO: check this

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    return model, optimizer, scheduler


def run_valid(args, model, valid_dataloader):
    logger.info("  ***** Running validation *****  *")
    # model.eval()   # do eval outside of this function
    valid_loss = []
    with torch.no_grad():
        for step, batch in tqdm.tqdm(enumerate(valid_dataloader)):
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'source_ids': batch[0],
                      'target_ids': batch[1],
                      'pseudo_ids': batch[2],
                      'num_source_tokens': batch[3],
                      'num_target_tokens': batch[4]}
            for v in inputs.values():
                if type(v) == torch.Tensor:
                    v.requires_grad = False
            
            loss = model(**inputs)
            valid_loss.append(loss.detach().cpu().numpy())
            # prevent grad overflow (no grad should be calculated after all)
            model.zero_grad()

    return np.mean(valid_loss)


def train(args, training_features, validation_features, model, tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0] and args.log_dir:
        tb_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        tb_writer = None

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
    else:
        amp = None

    # model recover
    recover_step = utils.get_max_epoch_model(args.output_dir)

    model.to(args.device)
    model, optimizer, scheduler = prepare_for_training(args, model, amp=amp)

    if args.n_gpu == 0 or args.no_cuda:
        per_node_train_batch_size = args.per_gpu_train_batch_size * args.gradient_accumulation_steps
    else:
        per_node_train_batch_size = args.per_gpu_train_batch_size * args.n_gpu * args.gradient_accumulation_steps

    train_batch_size = per_node_train_batch_size * (torch.distributed.get_world_size() if args.local_rank != -1 else 1)
    global_step = recover_step if recover_step else 0

    if args.num_training_steps == -1:
        args.num_training_steps = int(args.num_training_epochs * len(training_features) / train_batch_size)

    train_dataset = utils.Seq2seqDatasetForBert(
        features=training_features, max_source_len=args.max_source_seq_length,
        max_target_len=args.max_target_seq_length, vocab_size=tokenizer.vocab_size,
        cls_id=tokenizer.cls_token_id, sep_id=tokenizer.sep_token_id, pad_id=tokenizer.pad_token_id,
        mask_id=tokenizer.mask_token_id, random_prob=args.random_prob, keep_prob=args.keep_prob,
        offset=train_batch_size * global_step, num_training_instances=train_batch_size * args.num_training_steps,
    )
    valid_dataset = utils.Seq2seqDatasetForBert(
                        features=validation_features, max_source_len=args.max_source_seq_length,
                        max_target_len=args.max_target_seq_length, vocab_size=tokenizer.vocab_size,
                        cls_id=tokenizer.cls_token_id, sep_id=tokenizer.sep_token_id, pad_id=tokenizer.pad_token_id,
                        mask_id=tokenizer.mask_token_id, random_prob=args.random_prob, keep_prob=args.keep_prob,
                        offset=0, num_training_instances=len(validation_features),
                    )
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.valid_batch_size, collate_fn=utils.batch_list_to_batch_tensors)


    # logger.info("Check dataset:")
    # for i in range(5):
    #     source_ids, target_ids, pseudo_ids, num_source_tokens, num_target_tokens = train_dataset.__getitem__(i)
    #     logger.info("Instance-%d" % i)
    #     logger.info("Source tokens = %s" % " ".join(tokenizer.convert_ids_to_tokens(source_ids)))
    #     logger.info("Target tokens = %s" % " ".join(tokenizer.convert_ids_to_tokens(target_ids)))

    # logger.info("Mode = %s" % str(model))

    # Train!
    logger.info("  ***** Running training *****  *")
    logger.info("  Num examples = %d", len(training_features))
    logger.info("  Num Epochs = %.2f", len(train_dataset) / len(training_features))
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Batch size per node = %d", per_node_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", train_batch_size)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.num_training_steps)

    if args.num_training_steps <= global_step:
        logger.info("Training is done. Please use a new dir or clean this dir!")
    else:
        # The training features are shuffled
        train_sampler = SequentialSampler(train_dataset) \
            if args.local_rank == -1 else DistributedSampler(train_dataset, shuffle=False)
        train_dataloader = DataLoader(
            train_dataset, sampler=train_sampler,
            batch_size=per_node_train_batch_size // args.gradient_accumulation_steps,
            collate_fn=utils.batch_list_to_batch_tensors)

        train_iterator = tqdm.tqdm(
            train_dataloader, initial=global_step,
            desc="Iter (loss=X.XXX, lr=X.XXXXXXX)", disable=args.local_rank not in [-1, 0])

        model.train()
        model.zero_grad()

        tr_loss, logging_loss, best_valid_loss, best_checkpoints = 0.0, 0.0, 1e10, []

        for step, batch in enumerate(train_iterator):
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'source_ids': batch[0],
                      'target_ids': batch[1],
                      'pseudo_ids': batch[2],
                      'num_source_tokens': batch[3],
                      'num_target_tokens': batch[4]}
            loss = model(**inputs)
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training

            train_iterator.set_description('Iter (loss=%5.3f) lr=%9.7f' % (loss.item(), scheduler.get_lr()[0]))

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            logging_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logger.info("")
                    logger.info(" Step [%d ~ %d]: %.2f", global_step - args.logging_steps, global_step, logging_loss)
                    if tb_writer:
                        tb_writer.add_scalar("Loss/train", logging_loss, global_step)
                        tb_writer.add_scalar("LR", scheduler.get_lr()[0], global_step)
                    logging_loss = 0.0

                if args.local_rank in [-1, 0] and args.save_steps > 0 and \
                        (global_step % args.save_steps == 0 or global_step == args.num_training_steps):

                    # run validation                                        
                    model.eval()
                    
                    valid_loss = run_valid(args, model, valid_dataloader)
                    logger.info(f'Validation loss step {global_step}: {valid_loss}')
                    if best_valid_loss > valid_loss:
                        logger.info(f'Validation loss improved from {best_valid_loss} to {valid_loss}')
                        best_valid_loss = valid_loss
                    if tb_writer:
                        tb_writer.add_scalar("Loss/eval", valid_loss, global_step)

                    model.train()
                    model.zero_grad()
                    
                    save_path = os.path.join(args.output_dir, "ckpt-%d" % global_step)
                    os.makedirs(save_path, exist_ok=True)
                    model_to_save = model.module if hasattr(model, "module") else model
                    model_to_save.save_pretrained(save_path)
                    tokenizer.save_pretrained(save_path)

                    torch.save(args, os.path.join(save_path, "training_args.bin"))
                    logger.info("Saving model checkpoint %d into %s", global_step, save_path)

                    torch.save(optimizer.state_dict(), os.path.join(save_path, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(save_path, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", save_path)

                    # remove checkpoints above limit
                    if args.save_limit is not None:
                        best_checkpoints.append(["ckpt-%d" % global_step, valid_loss])
                        best_checkpoints.sort(key=(lambda x: x[1]))
                        remove_failed_checkpoints = []
                        if len(best_checkpoints) > args.save_limit:
                            for ckpt_dir, cur_loss in best_checkpoints[args.save_limit:]:
                                logger.info("Removing %s due to save limit", ckpt_dir)
                                try:
                                    shutil.rmtree(os.path.join(args.output_dir, ckpt_dir))
                                except:
                                    logger.info("Failed at removing %s", ckpt_dir)
                                    remove_failed_checkpoints.append([ckpt_dir, cur_loss])
                        best_checkpoints = best_checkpoints[:args.save_limit] + remove_failed_checkpoints
                        logger.info("Best checkpoints: {}".format(best_checkpoints))

    if args.local_rank in [-1, 0] and tb_writer:
        tb_writer.close()


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="Directory storing the training and validation data (used for caching)")
    parser.add_argument("--train_file", default=None, type=str, required=True,
                        help="Training data (json format) for training. Keys: source and target")
    parser.add_argument("--validation_file", default=None, type=str, required=True,
                        help="Training data (json format) for validation. Keys: source and target")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list:")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints and predictions will be written.")
    parser.add_argument("--log_dir", default=None, type=str,
                        help="The output directory where the log will be written.")

    ## Other parameters
    parser.add_argument("--config_name", default=None, type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default=None, type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default=None, type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")

    parser.add_argument("--max_source_seq_length", default=464, type=int,
                        help="The maximum total source sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--max_target_seq_length", default=48, type=int,
                        help="The maximum total target sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")

    parser.add_argument("--cached_valid_features_file", default=None, type=str,
                        help="Cached training features file")
    parser.add_argument("--cached_train_features_file", default=None, type=str,
                        help="Cached validatino features file")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--valid_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--label_smoothing", default=0.1, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_training_steps", default=-1, type=int,
                        help="set total number of training steps to perform")
    parser.add_argument("--num_training_epochs", default=10, type=int,
                        help="set total number of training epochs to perform (--num_training_steps has higher priority)")
    parser.add_argument("--num_warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument("--random_prob", default=0.1, type=float,
                        help="prob to random replace a masked token")
    parser.add_argument("--keep_prob", default=0.1, type=float,
                        help="prob to keep no change for a masked token")

    parser.add_argument('--logging_steps', type=int, default=500,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=1500,
                        help="Run validation and save checkpoint every X updates steps.")
    parser.add_argument('--save_limit', type=int, default=None,
                        help="Maximum number of saved checkpoints.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument("--workers", default=1, type=int)
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--reset_parameters', action='store_true', help="Do not use the pre-trained model parameters")
    args = parser.parse_args()
    return args


def prepare(args):
    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    os.makedirs(args.output_dir, exist_ok=True)
    json.dump(args.__dict__, open(os.path.join(
        args.output_dir, 'train_opt.json'), 'w'), sort_keys=True, indent=2)

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1

    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    logger.info("Training/evaluation parameters %s", args)

    # Before we do anything with models, we want to ensure that we get fp16 execution of torch.einsum if args.fp16 is set.
    # Otherwise it'll default to "promote" mode, and we'll get fp32 operations. Note that running `--fp16_opt_level="O2"`
    # will remove the need for this code, but it is still valid.
    if args.fp16:
        try:
            import apex
            apex.amp.register_half_function(torch, 'einsum')
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")


def get_model_and_tokenizer(args):
    config_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    # Hack to cope with updated version of Transformers API
    if args.model_type in ['minilm', 'unilm', 'xbert']:
        config_file = config_class.pretrained_config_archive_map[args.model_name_or_path]
        vocab_file = tokenizer_class.pretrained_vocab_files_map['vocab_file'][args.model_name_or_path]
        model_file = args.model_name_or_path
    elif args.model_type in ['minilm2-bert-base', 'minilm2-bert-large', 'minilm2-roberta']:
        args.tokenizer_name = TOKENIZER_NAMES[args.model_type]
        config_file = os.path.join(args.model_name_or_path, 'config.json')
        model_file = os.path.join(args.model_name_or_path, 'pytorch_model.bin')
    elif os.path.exists(args.model_name_or_path):
        vocab_file = os.path.join(args.model_name_or_path, 'vocab.txt')
        config_file = os.path.join(args.model_name_or_path, 'config.json')
        model_file = os.path.join(args.model_name_or_path, 'pytorch_model.bin')
        assert os.path.exists(vocab_file)
        assert os.path.exists(config_file)
        assert os.path.exists(model_file)
    else:
        vocab_file = args.model_name_or_path
        config_file = args.model_name_or_path
        model_file = args.model_name_or_path

    model_config = config_class.from_pretrained(
        args.config_name if args.config_name else config_file,
        cache_dir=args.cache_dir if args.cache_dir else None
    )

    config = BertForSeq2SeqConfig.from_exist_config(
        config=model_config, label_smoothing=args.label_smoothing,
        max_position_embeddings=args.max_source_seq_length + args.max_target_seq_length)
    logger.info("Model config for seq2seq: %s", str(config))
    
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else vocab_file,
        do_lower_case=args.do_lower_case, cache_dir=args.cache_dir if args.cache_dir else None
    )

    model = BertForSequenceToSequence.from_pretrained(
        model_file, config=config, model_type=args.model_type,
        reuse_position_embedding=True,
        cache_dir=args.cache_dir if args.cache_dir else None
    )

    # evaluate the model structure without pre-trained weights
    if args.reset_parameters:
        model.init_weights()

    total_model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Total model parameters: {}".format(total_model_params))

    return model, tokenizer


def main():
    args = get_args()
    prepare(args)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()
        # Make sure only the first process in distributed training will download model & vocab
    # Load pretrained model and tokenizer
    model, tokenizer = get_model_and_tokenizer(args)

    if args.local_rank == 0:
        torch.distributed.barrier()
        # Make sure only the first process in distributed training will download model & vocab

    if args.cached_train_features_file is None:
        args.cached_train_features_file = os.path.join(args.data_dir, f"cached_features_for_training_{args.model_type}.pt")
    if args.cached_valid_features_file is None:
        args.cached_valid_features_file = os.path.join(args.data_dir, f"cached_features_for_validation_{args.model_type}.pt")
    training_features = utils.load_and_cache_examples(
        example_file=args.train_file, tokenizer=tokenizer, local_rank=args.local_rank,
        cached_features_file=args.cached_train_features_file, shuffle=True, threads=args.workers
    )
    validation_features = utils.load_and_cache_examples(
        example_file=args.validation_file, tokenizer=tokenizer, local_rank=args.local_rank,
        cached_features_file=args.cached_valid_features_file, shuffle=False, threads=args.workers
    )

    train(args, training_features, validation_features, model, tokenizer)


if __name__ == "__main__":
    main()

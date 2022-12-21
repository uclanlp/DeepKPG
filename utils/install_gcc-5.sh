wget http://mirrors.concertpass.com/gcc/releases/gcc-5.5.0/gcc-5.5.0.tar.gz
tar xzf gcc-5.5.0.tar.gz
cd gcc-5.5.0
./contrib/download_prerequisites
cd ..
mkdir objdir
cd objdir
$PWD/../gcc-5.5.0/configure --prefix=$HOME/GCC-5.5.0 --enable-languages=c,c++,fortran,go
make
make install

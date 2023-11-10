FROM python

# Install system dependencies
# RUN gpg --keyserver hkp://keyserver.ubuntu.com:80 --recv 0E98404D386FA1D9 && gpg --export --armor 0E98404D386FA1D9 | sudo apt-key add -
# RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com --recv-keys 0E98404D386FA1D9 6ED0E7B82643E131 F8D2585B8783D481 0E98404D386FA1D9 6ED0E7B82643E131 54404762BBB6E853 BDE6D2B9216EC7A8
# RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 0E98404D386FA1D9
# RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 6ED0E7B82643E131
# RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys F8D2585B8783D481
# RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 0E98404D386FA1D9
# RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 6ED0E7B82643E131
# RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 54404762BBB6E853
# RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys BDE6D2B9216EC7A8
RUN apt-get update
RUN apt-get install -y gcc gfortran libcfitsio-dev libfftw3-dev libomp-dev libblas-dev liblapack-dev libgl1

# install fftw3
RUN wget http://www.fftw.org/fftw-3.3.10.tar.gz
RUN tar -xf fftw-3.3.10.tar.gz
WORKDIR /fftw-3.3.10
RUN ./configure --enable-openmp --enable-shared
RUN make
RUN make install

# copy the folder
WORKDIR /app
# ADD aicscytoparam ./aicscytoparam
WORKDIR /app

# Install Python dependencies
COPY setup.py .
COPY README.md .
# COPY setup.cfg .
RUN pip install --upgrade pip
RUN pip install "setuptools<60.0"
RUN pip install numpy
RUN pip install -e .
RUN pip install aicscytoparam
# RUN pip install cvapipe_analysis

# Copy the script and any other necessary files
COPY cellpack_analysis/scripts/cellPACK_PILR.py .

RUN ls -l

# Set the entrypoint script
COPY cellpack_analysis/scripts/run_cellPACK_PILR.sh ./run_cellPACK_PILR.sh
RUN chmod +x ./run_cellPACK_PILR.sh
ENTRYPOINT ["./run_cellPACK_PILR.sh"]
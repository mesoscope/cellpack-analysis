FROM python

# Install system dependencies
RUN apt-get update && apt-get install -y gcc gfortran libcfitsio-dev libfftw3-dev libomp-dev libblas-dev liblapack-dev libgl1

# install fftw3
RUN wget http://www.fftw.org/fftw-3.3.10.tar.gz
RUN tar -xf fftw-3.3.10.tar.gz
WORKDIR /fftw-3.3.10
RUN ./configure --enable-openmp --enable-shared
RUN make
RUN make install

# copy the folder
WORKDIR /app
ADD aicscytoparam ./aicscytoparam
WORKDIR /app

# Install Python dependencies
COPY setup.py .
COPY README.md .
COPY setup.cfg .
RUN pip install --upgrade pip
RUN pip install "setuptools<60.0"
RUN pip install numpy
RUN pip install -e .
RUN pip install aicscytoparam
RUN pip install cvapipe_analysis

# Copy the script and any other necessary files
COPY scripts/cellPACK_PILR.py .

RUN ls -l

# Set the entrypoint script
COPY scripts/run_cellPACK_PILR.sh ./run_cellPACK_PILR.sh
RUN chmod +x ./run_cellPACK_PILR.sh
ENTRYPOINT ["./run_cellPACK_PILR.sh"]
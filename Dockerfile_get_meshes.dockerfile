FROM python:3-bullseye

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

# Copy the script and any other necessary files
COPY scripts/get_meshes.py .

RUN ls -l

# Set the entrypoint script
COPY scripts/get_meshes.sh ./get_meshes.sh
RUN chmod +x ./get_meshes.sh
ENTRYPOINT ["./get_meshes.sh"]
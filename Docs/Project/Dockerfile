FROM continuumio/anaconda3
MAINTAINER "Brendan Magdamo"

RUN apt-get update && apt-get install -y libgtk2.0-dev && \
    rm -rf /var/lib/apt/lists/*

RUN /opt/conda/bin/conda update -n base -c defaults conda && \
    /opt/conda/bin/conda install python=3.6 && \
    /opt/conda/bin/conda install anaconda-client && \
    /opt/conda/bin/conda install jupyter -y && \
    /opt/conda/bin/conda install --channel https://conda.anaconda.org/menpo opencv3 -y && \
    /opt/conda/bin/conda install numpy pandas scikit-learn matplotlib seaborn pyyaml h5py keras xgboost torch -y && \
    /opt/conda/bin/conda upgrade dask && \

RUN ["mkdir", "notebooks"]
COPY conf/.jupyter /root/.jupyter
COPY run_jupyter.sh /

EXPOSE 8888

VOLUME /notebooks

CMD ["/run_jupyter.sh"]
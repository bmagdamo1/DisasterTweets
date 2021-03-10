mkdir notebooks 2> /dev/null;

docker build -t first_docker .;

docker run -it -p 8888:8888 -p 6006:6006 \
    -d -v $(pwd)/notebooks:/notebooks \
    python_data_science_container:anaconda;

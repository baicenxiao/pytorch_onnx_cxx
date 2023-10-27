# Toy example: Train pytorch model and export to ONNX. Perform inference via ONNX runtime in C++

1. Build docker container and run docker container:
    ```bash
    docker build -t onnxruntime_image .
    docker run -it --rm --name onnxruntime_container -v /path/on/host/to/onnx/model:/model onnxruntime_image
    ```
    If you are using MacOS with Apple Silicon chip, run the following build instead
    ```
    docker buildx build --platform linux/amd64 -t onnxruntime_image .
    ```

2. In docker container, compile one of the `test_modelx.cpp`
    ```
    g++ test_model3.cpp -o test_model3 -I/onnxruntime/onnxruntime-linux-x64-1.16.1/include -L/onnxruntime/onnxruntime-linux-x64-1.16.1/lib -lonnxruntime
    ```

3. Run the compiled file
    ```
    ./test_model3
    ```
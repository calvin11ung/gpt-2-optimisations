# Install dependencies
pip install -r requirements.txt

# Download webtext test set

python3 download_dataset.py

# Setup onnxruntime and quantize 8bit model

pip install onnxruntime
python3 onnx_runtime_quant.py 

# Build ggml, download GPT-2 XL model

git clone https://github.com/ggerganov/ggml.git

cd ggml

mkdir build && cd build

cmake ..

make -j4 gpt-2-backend gpt-2-quantize

bash ../examples/gpt-2/download-ggml-model.sh 1558M

./bin/gpt-2-quantize models/gpt-2-1558M/ggml-model.bin models/gpt-2-1558M/ggml-model-q4_0.bin 2

# Usage

python3 benchmark.py -p -b -n 10 -bsz 5

usage: benchmark.py [-h] [-b] [-p] [-v] [-s] [-n NUM_BATCHES] [-bsz BATCH_SIZE]

options:
  -h, --help            show this help message and exit

  -b, --benchmark       Run benchmarking

  -p, --perplexity      Calculate perplexity

  -v, --verbose         Print generated outputs

  -s, --sort_batches    Sort batches

  -n NUM_BATCHES, --num_batches NUM_BATCHES
                        Number of batches to test on
                        
  -bsz BATCH_SIZE, --batch_size BATCH_SIZE
                        Number of samples to run batched inference on

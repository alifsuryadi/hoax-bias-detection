from flask import Flask, request, jsonify, render_template
import os
import joblib  # Use joblib for loading pickle files
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences  # Use tf.keras

app = Flask(__name__)

# File paths
current_dir = os.path.dirname(os.path.abspath(__file__))

tokenizer_A3_path = os.path.join(current_dir, 'model', 'tokenizer_A3.pkl')
tokenizer_bias_path = os.path.join(current_dir, 'model', 'tokenizer_bias.pkl')

model_A3_path = os.path.join(current_dir, 'model', 'hoax_detection_A3.tflite')
model_bias_path = os.path.join(current_dir, 'model', 'bias_detection.tflite')

# Load tokenizer using joblib
def load_tokenizer(tokenizer_path):
    try:
        return joblib.load(tokenizer_path)
    except ModuleNotFoundError:
        # Handle missing module by injecting the correct module into sys.modules
        import tensorflow.keras.preprocessing.text
        import sys
        sys.modules['keras.src.preprocessing.text'] = tf.keras.preprocessing.text
        return joblib.load(tokenizer_path)

# Load the tokenizers
tokenizer_A3 = load_tokenizer(tokenizer_A3_path)
tokenizer_bias = load_tokenizer(tokenizer_bias_path)

# Load TFLite models
interpreterHoaks = tf.lite.Interpreter(model_path=model_A3_path)
interpreterHoaks.allocate_tensors()

interpreterBias = tf.lite.Interpreter(model_path=model_bias_path)
interpreterBias.allocate_tensors()

# Get input and output tensor information
input_details_hoaks = interpreterHoaks.get_input_details()
output_details_hoaks = interpreterHoaks.get_output_details()

input_details_bias = interpreterBias.get_input_details()
output_details_bias = interpreterBias.get_output_details()

@app.route('/')
def index():
    return render_template('deteksiHoaks.html')

@app.route('/deteksiBias')
def bias():
    return render_template('deteksiBias.html')

@app.route('/predictHoaks', methods=['POST'])
def predictHoaks():
    data = request.get_json()
    news_text1 = data['news_text1']
    news_text2 = data['news_text2']

    # Combine input texts
    news_text = [news_text1 + " " + news_text2]

    # Tokenization and padding of news
    new_sequences = tokenizer_A3.texts_to_sequences(news_text)
    max_len = 100
    new_padded = pad_sequences(new_sequences, maxlen=max_len)

    # Convert input data to float32 type
    new_padded = new_padded.astype('float32')

    # Set the input tensor with compacted data
    interpreterHoaks.set_tensor(input_details_hoaks[0]['index'], new_padded)

    # Run the interpreter to make predictions
    interpreterHoaks.invoke()

    # Get the prediction result from the output tensor
    predictions_tflite = interpreterHoaks.get_tensor(output_details_hoaks[0]['index'])

    # Interpreting prediction results
    predicted_labels_tflite = "Hoax" if predictions_tflite[0][0] > 0.5 else "Not Hoax"
    confidence = float(predictions_tflite[0][0])

    # Prepare response
    response = {
        'prediction': predicted_labels_tflite,
        'confidence': confidence
    }

    return jsonify(response)

@app.route('/predictBias', methods=['POST'])
def predictBias():
    data = request.get_json()
    news_text1 = data['news_text1']
    news_text2 = data['news_text2']

    # Combine input texts
    news_text = [news_text1 + " " + news_text2]

    # Tokenization and padding of news
    new_sequences = tokenizer_bias.texts_to_sequences(news_text)
    max_len = 30
    new_padded = pad_sequences(new_sequences, maxlen=max_len)

    # Convert input data to float32 type
    new_padded = new_padded.astype('float32')

    # Set the input tensor with compacted data
    interpreterBias.set_tensor(input_details_bias[0]['index'], new_padded)

    # Run the interpreter to make predictions
    interpreterBias.invoke()

    # Get the prediction result from the output tensor
    predictions_tflite = interpreterBias.get_tensor(output_details_bias[0]['index'])

    # Interpreting prediction results
    predicted_labels_tflite = "Bias" if predictions_tflite[0][0] > 0.5 else "Netral"
    confidence = float(predictions_tflite[0][0])

    # Prepare response
    response = {
        'prediction': predicted_labels_tflite,
        'confidence': confidence
    }

    return jsonify(response)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(debug=True, host='0.0.0.0', port=port)

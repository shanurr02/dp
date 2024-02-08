# # from flask import Flask, request, jsonify
# # import numpy as np
# # from PIL import Image
# # import easyocr
# # import joblib
# # import os
# # import pickle
# # import json
# # import pandas as pd
# # from werkzeug.utils import secure_filename

# # app = Flask(__name__)

# # UPLOAD_FOLDER = './uploads'
# # app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# # # Ensure the upload fold exists
# # os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# # # Load the model and vectorizer
# # model = pickle.load(open('dectree.pkl', 'rb'))
# # vectorizer = pickle.load(open('decvec.pkl', 'rb')) 
# # # script_dir = os.path.dirname(os.path.realpath(__file__))
# # # vector_path = os.path.join(script_dir, 'decvec.pkl')
# # # model_path = os.path.join(script_dir, 'dectree.pkl')
# # # vectorizer = joblib.load(vector_path)
# # # model = joblib.load(model_path)


# # # Initialize EasyOCR reader
# # reader = easyocr.Reader(['en'])  # Specify the language(s) you want to support

# # @app.route('/predict', methods=['POST'])
# # def predict_endpoint():
# #     if 'text' in request.data.decode('utf-8'):
# #         input_data = json.loads(request.data.decode('utf-8'))
# #         df = pd.DataFrame(input_data, columns=['id', 'text'])
# #         text_data = df['text'].values
# #         X_pred = vectorizer.transform(text_data)
# #         predictions = model.predict(X_pred)
# #         results_df = pd.DataFrame({'id': df['id'], 'prediction': predictions})
# #         return jsonify(results_df.to_dict(orient='records'))

# #     if 'image' in request.files:
# #         file = request.files['image']
# #         if file.filename == '':
# #             return jsonify({'error': 'No selected file'})
# #         # Save the uploaded file to the upload folder
# #         filename = secure_filename(file.filename)
# #         file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
# #         file.save(file_path)

# #         reader = easyocr.Reader(['en'])
# #         # Perform OCR on the image
# #         result = reader.readtext(file_path, paragraph=True)
# #         # Extract text from the result
# #         df = pd.DataFrame(result, columns=['bbox', 'text'])
# #         df['text'] = df['text'].apply(lambda x: x.lower())  # Convert text to lowercase
# #         text_data = df['text'].values
# #         X_pred = vectorizer.transform(text_data)
# #         predictions = model.predict(X_pred)
# #         results_df = pd.DataFrame({'prediction': predictions})
# #         return jsonify(results_df.to_dict(orient='records'))

# #     return jsonify({'error': 'No text or image data provided'}), 400

# # if __name__ == '__main__':
# #     app.run(debug=True)

# from flask import Flask, request, jsonify
# from funcd import find_matching_rows
# from funcd import create_final_dataframe
# import numpy as np
# from PIL import Image
# import easyocr
# import joblib
# import os
# import pickle
# import json
# import pandas as pd
# from werkzeug.utils import secure_filename

# app = Flask(__name__)

# UPLOAD_FOLDER = './uploads'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# # Ensure the upload folder exists
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# # Load the model and vectorizer
# model = pickle.load(open('dectree.pkl', 'rb'))
# vectorizer = pickle.load(open('decvec.pkl', 'rb')) 

# # Initialize EasyOCR reader
# # reader = easyocr.Reader(['en'])  # Specify the language(s) you want to support

# @app.route('/predict', methods=['POST'])
# def predict_endpoint():
#     if b'text' in request.data and 'image' in request.files:
#         input_data = json.loads(request.data)
#         print(input_data)
#         texts = pd.DataFrame.from_dict(input_data)
#         text_data = texts['text'].values
#         X_pred = vectorizer.transform(text_data)
#         predictions = model.predict(X_pred)
#         results_text = pd.DataFrame({ 'prediction': predictions})

#         # return jsonify(results_df.to_dict(orient='records'))

#     #
#         file = request.files['image']
#         if file.filename == '':
#             return jsonify({'error': 'No selected file'})
#         # Save the uploaded file to the upload folder
#         filename = secure_filename(file.filename)
#         file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(file_path)

#         reader = easyocr.Reader(['en'])
#         # Perform OCR on the image
#         result = reader.readtext(file_path, paragraph=True)
#         # Extract text from the result
#         img = pd.DataFrame(result)
#         img['text'] = img['text'].apply(lambda x: x.lower())  # Convert text to lowercase
#         text_data = img['text'].values
#         X_pred = vectorizer.transform(text_data)
#         predictions = model.predict(X_pred)
#         results_img = pd.DataFrame({'prediction': predictions})

#         # Combine both results_img and results_text
#         combined_results = {'image_predictions': results_img.to_dict(orient='records'), 'text_predictions': results_text.to_dict(orient='records')}
        
        
# #         matching_rows = find_matching_rows(results_img, results_text)

# #         # Create the final DataFrame
# #         final_df = create_final_dataframe(results_img, results_text, matching_rows)
# #         final = {'predictions' : final_df.to_dict(orient = 'records')}
# # # Return combined results as JSON
# #         return jsonify(final)
#         return jsonify(combined_results)
        
    
#         # return jsonify(results_img.to_dict(orient='records')) 

#     # Combine predictions for text and image data if needed

#     # Return the combined prediction result as JSON
#     return jsonify({'error': 'No text or image data provided'}), 400


# if __name__ == '__main__':
#     app.run(debug=True)


# shanur  

from flask import Flask, request, jsonify
import numpy as np
from funcd import create_final_dataframe
from funcd import find_matching_rows
import os
import pickle
import json
import pandas as pd
import easyocr
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the model and vectorizer
model = pickle.load(open('dectree.pkl', 'rb'))
vectorizer = pickle.load(open('decvec.pkl', 'rb'))

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    if 'text' in request.form and 'image' in request.files:
        text_data = json.loads(request.form['text'])
        image_file = request.files['image']

        # Process text data
        df_text = pd.DataFrame(text_data, columns=['id', 'text'])
        text_values = df_text['text'].values
        X_pred_text = vectorizer.transform(text_values)
        predictions_text = model.predict(X_pred_text)
        results_text = pd.DataFrame({'id': df_text['id'], 'text': df_text['text'], 'prediction_text': predictions_text})

        # Process image data
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(image_file.filename))
        image_file.save(image_path)

        # Use EasyOCR on the image
        reader = easyocr.Reader(['en'])
        result_image = reader.readtext(image_path, paragraph=True)
        df_image = pd.DataFrame(result_image, columns=['bbox', 'text'])
        df_image['text'] = df_image['text'].apply(lambda x: x.lower())
        text_values_image = df_image['text'].values
        X_pred_image = vectorizer.transform(text_values_image)
        predictions_image = model.predict(X_pred_image)
        results_image = pd.DataFrame({'id': df_text['id'], 'text': df_image['text'], 'prediction_image': predictions_image})

        matching_rows = find_matching_rows(results_image, results_text)

        #Create the final DataFrame
        final_df = create_final_dataframe(results_image, results_text, matching_rows)

        # Combine results
        combined_results = {
            'predictions': final_df.to_dict(orient='records')
        }

        return jsonify(combined_results)

    return jsonify({'error': 'No text or image data provided'}), 400

# if __name__ == '__main__':
#     app.run(debug=True)

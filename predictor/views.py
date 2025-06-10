import joblib
import numpy as np
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

# Load model dan transformer
model = joblib.load('ModelKNNForDiabet.joblib')
transformer = joblib.load('quantile_transformer.joblib')

def index(request):
    return render(request, 'predictor/index.html')

@csrf_exempt
def predict(request):
    if request.method == 'POST':
        try:
            # Ambil input dari form
            pregnancies = float(request.POST.get('pregnancies'))
            glucose = float(request.POST.get('glucose'))
            bmi = float(request.POST.get('bmi'))
            dpf = float(request.POST.get('dpf'))
            age = float(request.POST.get('age'))

            # Format data mentah
            data = np.array([[pregnancies, glucose, bmi, dpf, age]])

            # Transformasi fitur sesuai training
            transformed_data = transformer.transform(data)

            # Prediksi
            prediction = model.predict(transformed_data)
            probability = model.predict_proba(transformed_data)

            # Kirim response
            return JsonResponse({
                'prediction': int(prediction[0]),
                'probability': float(np.max(probability)) * 100,
                'status': 'success',
                'message': 'Prediksi berhasil dilakukan'
            })

        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': str(e)
            })

    return JsonResponse({
        'status': 'error',
        'message': 'Metode request tidak valid'
    })

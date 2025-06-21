import json
import os
from io import BytesIO

import chardet
from PIL import Image
from django.contrib import messages
from django.contrib.auth import authenticate, login
from django.contrib.auth import logout as auth_logout
from django.http import Http404
from django.http import HttpResponse
from django.http import HttpResponseRedirect, JsonResponse
from django.shortcuts import render, redirect
from django.urls import reverse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from openpyxl import Workbook
from openpyxl.drawing.image import Image as ExcelImage
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from .ML_Models import *
from .forms import SignupForm  # Import your SignupForm
from .models import AiModel, Result, AnalyseResult
from .models import RawDataset, PreprocessedDataset, DataVisualization , MyTrainedModels

# A dictionary mapping model names to their corresponding functions
MODEL_FUNCTIONS = {
    # "RNN":train_RNN,
    # "Random Forest": train_random_forest,
    # "Logistic Regression": train_logistic_regression,
    # "Random Forests": train_classification_random_forest,
    # "Support Vector Machines (SVC)": train_classification_svc,
    # "Support Vector Machines (SVR)": train_svr,

}


def login_required_custom(view_func):
    def _wrapped_view(request, *args, **kwargs):
        if not request.user.is_authenticated:
            return HttpResponseRedirect(reverse('page-login'))  # Redirect to login page
        return view_func(request, *args, **kwargs)

    return _wrapped_view


@login_required_custom
def logout(request):
    auth_logout(request)
    return redirect('page-login')


from django.contrib.auth.decorators import login_required
from django.db.models import Count

@login_required_custom
def my_view(request):
    raw, prep = getUploadedDatasets(request)
    modelsTrained = Result.objects.filter(utilisateur=request.user)
    dataSIZE = Patient.objects.count()

    myanalysisHystory = AnalyseResult.objects.filter(utilisateur=request.user)

    # Count predictions by value
    No_Stroke_Count = myanalysisHystory.filter(prediction="No Stroke").count()
    Stroke_Count = myanalysisHystory.filter(prediction="Stroke").count()

    context = {
        'dataUploaded': raw.count(),
        'preproccessedData': prep.count(),
        'modelsTrained': modelsTrained.count(),
        'FeaturesCount': 10,
        'dataSIZE': dataSIZE,
        "No_Stroke_Count": No_Stroke_Count,
        "Stroke_Count": Stroke_Count,
    }
    return render(request, 'Dashboard.html', context)

def page_login(request):
    if request.user.is_authenticated:
        return redirect('index')

    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']

        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect('index')
        else:
            message = 'Invalid username or password'
            return render(request, 'page-login.html', {'f': 'Invalid username or password'})

    return render(request, 'page-login.html')


def page_register(request):
    if request.user.is_authenticated:
        return redirect('index')

    elif request.method == 'POST':
        form = SignupForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            messages.success(request, 'Account created successfully! You can now log in.')
            return redirect('page-login')
        else:
            print(form.errors)  # Add this to log the errors

            messages.error(request, 'There were errors in your form. Please check.')

    else:
        form = SignupForm(initial={
            'username': '',
            'email': '',
            'profile_picture_path': None,
            'password1': '',
            'password2': '',
            'country': '',
            'full_name': '',

        })

    return render(request, 'page-register.html', {'form': form})


@login_required_custom
def chart_peity(request):
    return render(request, 'chart-peity.html')


@login_required_custom
def chart_sparkline(request):
    return render(request, 'chart-sparkline.html')


@login_required_custom
def chart_chartist(request):
    return render(request, 'chart-chartist.html')


@login_required_custom
def chart_chartjs(request):
    return render(request, 'chart-chartjs.html')


@login_required_custom
def chart_morris(request):
    return render(request, 'chart-morris.html')


@login_required_custom
def chart_flot(request):
    return render(request, 'chart-flot.html')


@login_required_custom
def tablePage(request):
    return render(request, 'tablePage.html')


@login_required_custom
def tableData(request):
    return render(request, 'table-datatable.html')


@login_required_custom
def app_profile(request):
    return render(request, 'app-profile.html')


@csrf_exempt
@login_required_custom
def uploadDataFile(request):
    if request.method == 'POST':
        uploaded_file = request.FILES.get('file')
        target_column = request.POST.get('target_column')  # Optional for unsupervised
        custom_name = request.POST.get('custom_name')
        skip_preprocessing = request.POST.get('skip_preprocessing')
        visibility = request.POST.get('visibility')

        utilisateur = request.user  # Assuming the user is authenticated

        # Validate inputs
        if not uploaded_file:
            return JsonResponse({'error': 'File is required.'}, status=400)
        if not custom_name:
            return JsonResponse({'error': 'Dataset custom name is required.'}, status=400)

        # Validate file type (accepting both CSV and Excel)
        if not (uploaded_file.name.endswith('.csv') or uploaded_file.name.endswith(
                '.xlsx') or uploaded_file.name.endswith('.xls')):
            return JsonResponse({'error': 'Only CSV and Excel files are supported.'}, status=400)

        try:
            # Handle CSV and Excel files
            if uploaded_file.name.endswith('.csv'):
                # Detect file encoding
                raw_data = uploaded_file.read(1000)
                result = chardet.detect(raw_data)
                encoding = result['encoding'] or 'ISO-8859-1'  # Default to 'ISO-8859-1' if detection fails
                uploaded_file.seek(0)  # Reset file pointer after reading
                df = pd.read_csv(uploaded_file, encoding=encoding)
            elif uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file, engine='openpyxl')  # For newer Excel files
            elif uploaded_file.name.endswith('.xls'):
                df = pd.read_excel(uploaded_file, engine='xlrd')  # For older Excel files

            # Check if the dataset is empty
            if df.empty:
                return JsonResponse({'error': 'The dataset is empty.'}, status=400)

            # For supervised learning, validate target column
            if target_column:
                if target_column not in df.columns:
                    target_column = target_column.replace('"', '')
                    if target_column not in df.columns:
                        return JsonResponse(
                            {'error': f'Target column "{target_column}" not found in the uploaded file.'}, status=400)

            # Save the file and dataset details using the RawDataset model
            raw_dataset = RawDataset.objects.create(
                utilisateur=utilisateur,
                file_raw_dataset=uploaded_file,
                TargetColumn=target_column if target_column else None,
                datasetCostumName=custom_name,
                visibility=visibility,
            )

            file_path = raw_dataset.file_raw_dataset.path
            if not os.path.exists(file_path):
                return JsonResponse({'error': f'File path {file_path} does not exist.'}, status=500)
            # Process the data
            # add file path to the preprocessed dataset
            preprocessed_dataset = PreprocessedDataset.objects.create(
                raw_dataset=raw_dataset,
                preprocessedCostumName=custom_name + '_preprocessed',
                file_preprocessed_data=uploaded_file,
                visibility=visibility,
            )



            if target_column and skip_preprocessing == 'false':
                # Supervised workflow
                try:
                    X_train, X_test, y_train, y_test = preprocessed_dataset.process_data(target_column)
                except Exception as e:
                    print(f"Error during supervised data processing: {e}")
                    return JsonResponse({'error': 'An error occurred during data preprocessing.'}, status=500)

            elif not target_column and skip_preprocessing == 'false':
                # Unsupervised workflow
                try:
                    processed_data = preprocessed_dataset.process_data_unsupervised()
                except Exception as e:
                    print(f"Error during unsupervised data processing: {e}")
                    return JsonResponse({'error': 'An error occurred during unsupervised data processing.'}, status=500)

            # Return a success response
            return JsonResponse({'message': 'File uploaded and processed successfully!',
                                 'preprocessing_url': '/uploadedFiles'}, status=200)

        except Exception as e:
            # Log the exception for debugging
            print(f"Error processing file: {e}")
            raise e
            return JsonResponse({'error': f'An error occurred while processing the file: {str(e)}'}, status=500)

    return render(request, 'uploadDataFile.html')

#
# def getUploadedDatasets(request):
#     uploadedfiles = RawDataset.objects.filter(utilisateur=request.user).order_by('-id')
#     processeddatasets = PreprocessedDataset.objects.filter(raw_dataset__utilisateur=request.user).order_by('-id')
#     return uploadedfiles, processeddatasets
#

def getUploadedDatasets(request):
    user = request.user

    # Fetch user's own datasets
    user_datasets = RawDataset.objects.filter(utilisateur=user).order_by('-id')

    # Fetch public datasets uploaded by other users
    public_datasets = RawDataset.objects.filter(visibility='Public').exclude(utilisateur=user).order_by('-id')

    # Combine both querysets
    uploadedfiles = user_datasets | public_datasets

    # Processed datasets uploaded by the user
    userprocesseddatasets = PreprocessedDataset.objects.filter(raw_dataset__utilisateur=user).order_by('-id')
    publicproccesseddatasets = PreprocessedDataset.objects.filter(visibility='Public').exclude(raw_dataset__utilisateur=user).order_by('-id')
    processeddatasets = userprocesseddatasets | publicproccesseddatasets

    return uploadedfiles, processeddatasets


@login_required_custom
def uploadedFiles(request):
    uploadedfiles, processeddatasets = getUploadedDatasets(request)
    form = {'uploadedfiles': uploadedfiles, 'processeddatasets': processeddatasets}
    return render(request, 'uploadedFiles.html', form)


@login_required_custom
def AVC_Detection(request):
    uploadedfiles, processeddatasets = getUploadedDatasets(request)
    form = {'uploadedfiles': uploadedfiles, 'processeddatasets': processeddatasets}

    return render(request, 'classification.html', form)


@login_required_custom
def Risk_Score(request):
    uploadedfiles, processeddatasets = getUploadedDatasets(request)
    form = {'uploadedfiles': uploadedfiles, 'processeddatasets': processeddatasets}

    return render(request, 'regression.html', form)



@login_required_custom
def mytrainedmodels(request):
    MyTrainedModelss = MyTrainedModels.objects.filter(utilisateur=request.user).order_by('-id')
    form = {'MyTrainedModels': MyTrainedModelss}

    return render(request, 'mytrainedmodels.html', form)


@login_required_custom
def clustering(request):
    uploadedfiles, processeddatasets = getUploadedDatasets(request)
    form = {'uploadedfiles': uploadedfiles, 'processeddatasets': processeddatasets}

    return render(request, 'clustering.html', form)


@login_required_custom
@csrf_exempt
def train_model_view(request, model_name, processed_file_id, supervised):
    if request.method == "POST":
        # Extract 'params' from the POST body
        params = request.POST.get("params")  # Use this if the data is form-encoded
        params_dict = json.loads(params)

        # Fetch the PreprocessedDataset object
        preprocessed_dataset = PreprocessedDataset.objects.get(id=processed_file_id)

        # Fetch the target column from the associated RawDataset
        target_column = preprocessed_dataset.raw_dataset.TargetColumn

        if supervised == "supervised":
            processedData = preprocessed_dataset.process_data(target_column)
        else:
            target_column = None
            processedData = preprocessed_dataset.process_data_unsupervised()

        # Check if the selected model exists in the mapping
        model_function = MODEL_FUNCTIONS.get(model_name)

        context = {
            "message": "Couldnt Start traning , Invalid model or dataset !",
            "status": 'failed',
            "result": 'no result to show',

            "modelName": model_name,
            "dataCostumName": preprocessed_dataset.raw_dataset.datasetCostumName
        }

        try:
            dsCname = preprocessed_dataset.raw_dataset.datasetCostumName
            # Execute the associated function, passing the file path and target column
            result = model_function(datasetCostumName=dsCname,authenticatedUser=request.user)
            # Check if result contains DataFrame and convert it to a serializable format
            if isinstance(result, pd.DataFrame):
                result = result.to_dict(orient='records')  # Convert DataFrame to list of dictionaries

            context = {
                "message": f"Model trained successfully",
                "result": result,
                "status": 'success',
                "modelName": model_name,
                "dataCostumName": preprocessed_dataset.raw_dataset.datasetCostumName
            }
            print(f'context:\n {context}')
            # Serialize the context into a JSON string
            context_json = json.dumps(context,default=str)

            # Check if the model already exists
            mlmodel = AiModel.objects.filter(name=model_name, model_params=params_dict).first()

            if mlmodel:
                # Create a new Result entry if the model exists
                Result.objects.create(
                    utilisateur=request.user,
                    ai_model=mlmodel,
                    preprocessed_dataset=preprocessed_dataset,
                    resultobject=context_json  # Store the serialized JSON string
                )
            else:
                # Create the AiModel and associate it with a new Result entry
                mlmodel = AiModel.objects.create(
                    name=model_name,
                    model_params=params_dict
                )
                Result.objects.create(
                    utilisateur=request.user,
                    ai_model=mlmodel,
                    preprocessed_dataset=preprocessed_dataset,
                    resultobject=context_json  # Store the serialized JSON string
                )
        except Exception as e:

            context = {
                "message": f"Error during training",
                "result": f'{str(e)}',
                "status": 'failed',
                "modelName": model_name,
                "dataCostumName": preprocessed_dataset.raw_dataset.datasetCostumName
            }
            # raise e

        return render(request, 'train_result.html', context)


def visualize_data(request, datatype, dataset_id):
    """
    Vue pour afficher les visualisations associées à un jeu de données brut ou prétraité.
    """
    if datatype == 'raw':
        # Recherche dans RawDataset
        dataset = RawDataset.objects.filter(id=dataset_id).first()
        data_visualizations = DataVisualization.objects.filter(dataset=dataset, dataset_processed__isnull=True)

        if not data_visualizations.exists():
            dataset.generate_visualizations()
            data_visualizations = DataVisualization.objects.filter(dataset=dataset, dataset_processed__isnull=True)

        return render(request, 'visualisationData.html', {
            'data_visualizations': data_visualizations,
            'dataset': dataset,
        })

    else:
        dataset = PreprocessedDataset.objects.filter(id=dataset_id).first()
        data_visualizations = DataVisualization.objects.filter(dataset_processed=dataset)

        if not data_visualizations.exists():
            dataset.generate_visualizations()
            data_visualizations = DataVisualization.objects.filter(dataset_processed=dataset)

        return render(request, 'visualisationData.html', {
            'data_visualizations': data_visualizations,
            'dataset': dataset,
        })


@login_required_custom
def Results(request):
    results = Result.objects.filter(utilisateur=request.user).order_by('-id')
    return render(request, 'modelsresults.html', {'results': results})


def visualize_result(request, resultID):
    result = Result.objects.filter(id=resultID).first()
    context = json.loads(result.resultobject)
    print(f'returning context as {context}')
    return render(request, 'train_result.html', context)


def downloadPreproccesseddata(request, prepdataID):
    # Retrieve the dataset object
    dataset = PreprocessedDataset.objects.filter(id=prepdataID).first()

    # Check if the dataset and file exist
    if dataset and dataset.file_preprocessed_data:
        # Assuming `file_preprocessed_data` is a FileField or similar
        file_path = dataset.file_preprocessed_data.path  # Get the file's path

        try:
            with open(file_path, 'rb') as file:
                response = HttpResponse(file.read(), content_type='application/octet-stream')
                response['Content-Disposition'] = f'attachment; filename="{dataset.file_preprocessed_data.name}"'
                return response
        except FileNotFoundError:
            raise Http404("File not found.")

    # Redirect to uploadedFiles page if the file or dataset is not found
    return redirect('uploadedFiles')


def download_report(request, resultID):
    # Fetch the result object
    result = Result.objects.filter(id=resultID).first()
    context = json.loads(result.resultobject)

    # Prepare the PDF response
    buffer = BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    # Set font and size
    p.setFont("Helvetica", 23)

    # Add content to the PDF
    p.drawString(115, height - 50, f" NEUROLAB -  Model Training Report ")
    p.drawString(50, height - 50, f"")
    p.drawString(50, height - 50, f"")
    p.drawString(50, height - 50, f"")
    p.drawString(50, height - 50, f"")

    p.setFont("Helvetica-Bold", 18)
    p.drawString(50, height - 120, f"Model Name: {context.get('modelName', 'N/A')}")
    p.drawString(50, height - 150, f"Dataset: {context.get('dataCostumName', 'N/A')}")

    # Add Metrics
    p.setFont("Helvetica-Bold", 16)
    p.drawString(50, height - 180, "Metrics:")
    y = height - 200
    p.setFont("Helvetica", 10)
    for metric, value in context['result']['metric_results'].items():
        p.drawString(70, y, f"{metric}: {value}")
        y -= 20

    # Add Plots to the PDF, 2 plots per page
    if context['result'].get('plots'):
        plot_count = 0
        p.showPage()  # Start with a new page for plots
        p.setFont("Helvetica-Bold", 12)
        p.drawString(50, height - 50, "Plots:")
        y = height - 80
        p.setFont("Helvetica", 10)

        max_width = width * 0.7  # 70% of page width
        max_height = height * 0.35  # 35% of page height

        for plot_key, plot_value in context['result']['plots'].items():
            # Decode the Base64 image data
            image_data = base64.b64decode(plot_value)
            image_file = BytesIO(image_data)

            # Use PIL to open the image from BytesIO
            image = Image.open(image_file)

            # Save the image temporarily to a file
            temp_file_path = f"/tmp/plot_{plot_key}.png"
            image.save(temp_file_path)

            # Calculate the position to center the image
            x_pos = (width - max_width) / 2  # Horizontal center
            y_pos = y - max_height  # Adjust Y for the image position

            # Insert the plot image into the PDF
            # p.drawString(x_pos, y+10, f"{plot_key}:")
            p.drawImage(temp_file_path, x_pos, y_pos, width=max_width,
                        height=max_height)  # Adjust image size and positioning
            y -= max_height + 50  # Adjust spacing for next plot

            # After two plots, add a new page
            plot_count += 1
            if plot_count % 2 == 0:
                p.showPage()  # Start a new page for the next plot set
                p.setFont("Helvetica", 10)
                p.drawString(50, height - 50, "Plots:")
                y = height - 70

            # Clean up the temporary file after use
            os.remove(temp_file_path)

    # Close the PDF
    p.showPage()
    p.save()

    # Create the response
    buffer.seek(0)
    response = HttpResponse(buffer, content_type='application/pdf')
    response['Content-Disposition'] = f'attachment; filename="report_{resultID}.pdf"'
    return response


def download_excel(request, resultID):
    # Fetch the result object
    result = Result.objects.filter(id=resultID).first()
    context = json.loads(result.resultobject)

    # Create an Excel workbook
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Model Report"

    # Add model info
    sheet["A1"] = "Model Report"
    sheet["A3"] = "Model Name:"
    sheet["B3"] = context.get('modelName', 'N/A')
    sheet["A4"] = "Dataset:"
    sheet["B4"] = context.get('dataCostumName', 'N/A')

    # Add metrics
    sheet["A6"] = "Metrics"
    row = 7
    for metric, value in context['result']['metric_results'].items():
        sheet[f"A{row}"] = metric
        sheet[f"B{row}"] = value
        row += 1

    # Add plots
    if context['result'].get('plots'):
        sheet["A{row}".format(row=row + 2)] = "Plots"
        row += 3
        for plot_key, plot_value in context['result']['plots'].items():
            # Decode Base64 image
            image_data = base64.b64decode(plot_value)
            image_file = BytesIO(image_data)
            img = ExcelImage(image_file)

            # Add image and label to sheet
            sheet[f"A{row}"] = plot_key
            img.anchor = f"A{row + 1}"  # Position of the image
            sheet.add_image(img)
            row += 35  # Adjust for image height

    # Save the workbook to a BytesIO stream
    output = BytesIO()
    workbook.save(output)
    output.seek(0)

    # Return as Excel file response
    response = HttpResponse(output, content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    response['Content-Disposition'] = f'attachment; filename="report_{resultID}.xlsx"'
    return response


def home(request):
    return render(request, 'homePage.html')


import os
from django.shortcuts import get_object_or_404, redirect
from django.conf import settings


def deleteData(request, prepdataID):
    # Get the preprocessed dataset instance
    preprocessed_dataset = get_object_or_404(PreprocessedDataset, id=prepdataID)

    # Get the associated raw dataset
    raw_dataset = preprocessed_dataset.raw_dataset

    # Delete preprocessed file from storage
    if preprocessed_dataset.file_preprocessed_data:
        file_path = os.path.join(settings.MEDIA_ROOT, str(preprocessed_dataset.file_preprocessed_data))
        if os.path.exists(file_path):
            os.remove(file_path)

    # Delete preprocessed dataset entry
    preprocessed_dataset.delete()

    # Check if there are any other preprocessed datasets linked to this raw dataset
    remaining_preprocessed = PreprocessedDataset.objects.filter(raw_dataset=raw_dataset).exists()

    # If no more preprocessed datasets exist, delete the raw dataset
    if not remaining_preprocessed:
        if raw_dataset.file_raw_dataset:
            raw_file_path = os.path.join(settings.MEDIA_ROOT, str(raw_dataset.file_raw_dataset))
            if os.path.exists(raw_file_path):
                os.remove(raw_file_path)
        raw_dataset.delete()

    return redirect('uploadedFiles')

from .models import MyTrainedModels  # adjust import if needed
from keras.models import load_model
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from .models import MyTrainedModels

def predict(request):
    return render(request, 'avc_predict.html')

def insertPage(request):
    return render(request, 'avc_insert_patient.html',)

from django.utils.timezone import now
from .ML_Models import *  # replace with correct import


def InsertPatientData(request):
    try:
        if request.method == 'POST':
            user_input = {
                "gender": request.POST.get("gender"),
                "age": float(request.POST.get("age")),
                "hypertension": int(request.POST.get("hypertension")),
                "heart_disease": int(request.POST.get("heart_disease")),
                "ever_married": request.POST.get("ever_married"),
                "work_type": request.POST.get("work_type"),
                "residence_type": request.POST.get("Residence_type"),
                "avg_glucose_level": float(request.POST.get("avg_glucose_level")),
                "bmi": float(request.POST.get("bmi")),
                "smoking_status": request.POST.get("smoking_status"),
                "stroke": int(request.POST.get("stroke"))
            }

            # Insert patient into database
            Patient.objects.create(**user_input)

            model_name = request.POST.get('model_choice')

            action = request.POST.get('action')
            if action == "insert_and_train":
                if model_name == "RNN":
                    result=train_RNN_2(authenticatedUser=request.user)
                elif model_name == "LogisticRegression":
                    result=train_logistic_regression(authenticatedUser=request.user)
                elif model_name == "RandomForest":
                    result=train_classification_random_forest(authenticatedUser=request.user)
                elif model_name == "SVM":
                    result=train_classification_svm_classification(authenticatedUser=request.user)
                elif model_name == "LightGBM":
                    result=train_lightgbm(authenticatedUser=request.user)

                print("✅ Model retrained after insertion.")

                context = {
                    "message": f"Model trained successfully",
                    "result": result,
                    "status": 'success',
                    "modelName": model_name,
                    "dataCostumName": "Stroke"
                }
                print(f'context:\n {context}')
                # Serialize the context into a JSON string
                context_json = json.dumps(context, default=str)

                # Check if the model already exists
                mlmodel = AiModel.objects.filter(name=model_name, model_params=None).first()

                if mlmodel:
                    # Create a new Result entry if the model exists
                    Result.objects.create(
                        utilisateur=request.user,
                        ai_model=mlmodel,
                        # preprocessed_dataset=preprocessed_dataset,
                        resultobject=context_json  # Store the serialized JSON string
                    )
                else:
                    # Create the AiModel and associate it with a new Result entry
                    mlmodel = AiModel.objects.create(
                        name=model_name,
                        model_params={}
                    )
                    Result.objects.create(
                        utilisateur=request.user,
                        ai_model=mlmodel,
                        # preprocessed_dataset=preprocessed_dataset,
                        resultobject=context_json  # Store the serialized JSON string
                    )
                return render(request, 'train_result.html',context)

            else:
                print("✅ Data inserted without model training.")
                return render(request, 'avc_insert_patient.html')

    except Exception as e:
        print(f"Insertion patient data Exception : {e}")
        raise e
        return render(request, 'avc_insert_patient.html')



def load_any_model(path):
    ext = os.path.splitext(path)[-1].lower()

    if ext == ".pkl":
        return joblib.load(path)
    elif ext in [".h5", ".keras"]:
        return load_model(path)
    elif ext == ".txt":
        return lgb.Booster(model_file=path)
    else:
        raise ValueError("Unsupported model format:", ext)


def classification(request):
    return render(request,"classification.html")

@csrf_exempt
def train_modell_view(request, model_name):
    try:
        if request.method == "POST":
            user = request.user
            result = None

            if model_name == "RNN":
                result = train_RNN_2(authenticatedUser=user)
            elif model_name == "Logistic Regression":
                result = train_logistic_regression(authenticatedUser=user)
            elif model_name == "Random Forests":
                result = train_classification_random_forest(authenticatedUser=user)
            elif model_name == "Support Vector Machines (SVC)":
                result = train_classification_svm_classification(authenticatedUser=user)
            elif model_name == "LightGBM":
                result = train_lightgbm(authenticatedUser=user)
            else:
                raise Exception("Unknown model")

            context = {
                "message": f"Model {model_name} trained successfully.",
                "result": result,
                "status": "success",
                "modelName": model_name,
                "dataCostumName": "Stroke"
            }
            context_json = json.dumps(context, default=str)

            # save Result in DB
            mlmodel = AiModel.objects.filter(name=model_name).first()
            if mlmodel:
                # Create a new Result entry if the model exists
                Result.objects.create(
                    utilisateur=request.user,
                    ai_model=mlmodel,
                    # preprocessed_dataset=preprocessed_dataset,
                    resultobject=context_json  # Store the serialized JSON string
                )
            else:
                # Create the AiModel and associate it with a new Result entry
                mlmodel = AiModel.objects.create(
                    name=model_name,
                    model_params={}
                )
                Result.objects.create(
                    utilisateur=request.user,
                    ai_model=mlmodel,
                    # preprocessed_dataset=preprocessed_dataset,
                    resultobject=context_json  # Store the serialized JSON string
                )
            return render(request, 'train_result.html', context)

    except Exception as e:
        raise e
        context = {
            "message": f"Error during training",
            "result": f'{str(e)}',
            "status": 'failed',
            "modelName": model_name,
        }
        return render(request, 'train_result.html', context)


def Prediction(request):
    try:
        if request.method == 'POST':
            user_input = {
                "gender": request.POST.get("gender"),
                "age": float(request.POST.get("age")),
                "hypertension": int(request.POST.get("hypertension")),
                "heart_disease": int(request.POST.get("heart_disease")),
                "ever_married": request.POST.get("ever_married"),
                "work_type": request.POST.get("work_type"),
                "residence_type": request.POST.get("Residence_type"),
                "avg_glucose_level": float(request.POST.get("avg_glucose_level")),
                "bmi": float(request.POST.get("bmi")),
                "smoking_status": request.POST.get("smoking_status")
            }

            print("\n🔮 Predicting stroke for user input data ...\n")
            print(user_input)

            # Load latest trained model
            trained_model_instance = MyTrainedModels.objects.latest('trained_at')

            model_path = trained_model_instance.modelFile.path
            print(f'📂 Using model from: {model_path}')

            # Load model
            model = load_any_model(model_path)

            # Load saved encoders and scaler
            import pickle
            with open('media/trained_models/label_encoders.pkl', 'rb') as f:
                label_encoders_dict = pickle.load(f)
            with open('media/trained_models/scaler.pkl', 'rb') as f:
                scaler = pickle.load(f)

            # Convert user input to DataFrame
            user_df = pd.DataFrame([user_input])

            # Apply encoding
            for col, le in label_encoders_dict.items():
                if col in user_df.columns:
                    user_df[col] = le.transform(user_df[col].astype(str))

            # Scale
            user_scaled = scaler.transform(user_df)

            # Check if it's an RNN model
            is_rnn_model = model_path.endswith(".keras")

            if is_rnn_model:
                user_input_array = user_scaled.reshape(1, 1, user_scaled.shape[1])
            else:
                user_input_array = user_scaled

            # Predict
            pred_result = model.predict(user_input_array)

            # Get probability
            if is_rnn_model:
                pred_prob = float(pred_result[0][0])
            elif hasattr(model, 'predict_proba'):
                pred_prob = float(model.predict_proba(user_input_array)[0][1])
            else:
                pred_prob = float(pred_result[0])  # fallback

            # Determine label
            pred_label = "Stroke" if pred_prob > 0.5 else "No Stroke"

            print(f"model name : {trained_model_instance.modelName}: 🧍 Predicted: {pred_label} (probability = {pred_prob:.4f})")
            try:
                AnalyseResult.objects.create(
                    utilisateur=request.user,
                    model_name=trained_model_instance.modelName,
                    prediction=pred_label,
                    proba=round(pred_prob * 100, 2),
                    input_data=user_input
                )
                print("analyse results saved to db successfully")
            except Exception as e :
                pass

            context = {
                "status": "success",
                "Predicted": pred_label,
                "probability": round(pred_prob * 100, 2),
                "userinp": user_input
            }
            return render(request, 'avc_prediction_result.html', context)

        return render(request, 'avc_predict.html')

    except Exception as e:
        print("❌ Prediction failed:", str(e))
        context = {
            "status": "failed",
            "result": f"❌ Prediction failed: {str(e)}",
        }
        return render(request, 'avc_prediction_result.html', context)



def viewpredictionHisotry(request):
    context = {
        "results":AnalyseResult.objects.filter(utilisateur=request.user).order_by('-id')
    }
    return render(request,"prediction_results.html", context)



#
#
# import json
# import requests
# from django.http import JsonResponse
# from django.views.decorators.csrf import csrf_exempt
# from django.views.decorators.http import require_http_methods
#
# # Replace with your Hugging Face API token here:
# HF_API_TOKEN = "******"
#
# API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-small"
# HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}
#
# def query_hf_api(payload):
#     print("Sending request to Hugging Face API...")
#     try:
#         response = requests.post(API_URL, headers=HEADERS, json=payload)
#         print(f"HTTP status code: {response.status_code}")
#         response.raise_for_status()  # raise error for bad responses
#         json_response = response.json()
#         print(f"Response JSON: {json_response}")
#         return json_response
#     except requests.exceptions.RequestException as e:
#         print(f"Request to HF API failed: {e}")
#         raise
#
# @csrf_exempt
# @require_http_methods(["POST"])
# def generate_health_advice(request):
#     print("Received request to generate health advice")
#     try:
#         data = json.loads(request.body)
#         print(f"Request data: {data}")
#
#         age = int(float(data.get('age', 0)))
#         bmi = float(data.get('bmi', 0))
#         glucose = float(data.get('glucose_level', 0))
#         smoking = data.get('smoking_status', 'Unknown')
#         gender = data.get('gender', 'Unknown')
#         hypertension = data.get('hypertension', '0')
#         heart_disease = data.get('heart_disease', '0')
#         risk_percentage = float(data.get('risk_percentage', 0.0))
#
#         print(f"Parsed inputs - age: {age}, bmi: {bmi}, glucose: {glucose}, smoking: {smoking}, gender: {gender}, hypertension: {hypertension}, heart_disease: {heart_disease}, risk_percentage: {risk_percentage}")
#
#         if bmi < 18.5:
#             bmi_category = "underweight"
#         elif bmi < 25:
#             bmi_category = "normal weight"
#         elif bmi < 30:
#             bmi_category = "overweight"
#         else:
#             bmi_category = "obese"
#         print(f"BMI category: {bmi_category}")
#
#         if glucose < 100:
#             glucose_status = "normal"
#         elif glucose < 126:
#             glucose_status = "prediabetic range"
#         else:
#             glucose_status = "diabetic range"
#         print(f"Glucose status: {glucose_status}")
#
#         prompt = f"""As a health advisor, provide personalized health recommendations for a {age}-year-old {gender.lower()} with the following health profile:
#
# Health Metrics:
# - BMI: {bmi} ({bmi_category})
# - Average Glucose Level: {glucose} mg/dL ({glucose_status})
# - Smoking Status: {smoking}
# - Hypertension: {"Yes" if hypertension == "1" else "No"}
# - Heart Disease: {"Yes" if heart_disease == "1" else "No"}
# - Stroke Risk: {risk_percentage}%
#
# Please provide:
# 1. An assessment of their current health status
# 2. 3-4 specific, actionable recommendations to improve or maintain their health
# 3. Lifestyle modifications that could help reduce stroke risk
# 4. When to consult healthcare professionals
#
# Format the response in HTML with proper headings and bullet points. Keep it encouraging and practical.
# """
#         print("Prompt to send to HF API:")
#         print(prompt)
#
#         hf_response = query_hf_api({"inputs": prompt})
#
#         print("Hugging Face API response received.")
#         print(f"Raw response: {hf_response}")
#
#         advice = hf_response[0].get('generated_text', '')
#         print(f"Extracted advice: {advice}")
#
#         return JsonResponse({'success': True, 'advice': advice})
#
#     except Exception as e:
#         print(f"Exception caught in generate_health_advice: {e}")
#         return JsonResponse({'success': False, 'error': str(e)}, status=500)

#
# import json
# import os
#
# from django.http import JsonResponse
# from django.shortcuts import HttpResponse
#
# import openai
#
# @csrf_exempt
# @require_http_methods(["POST"])
# def generate_health_advice(request):
#     print('Generating health advice...')
#     try:
#         data = json.loads(request.body)
#         print("Raw request body parsed.", data)
#
#         # Extract and validate user data
#         age = int(float(data.get('age', 0)))
#         bmi = float(data.get('bmi', 0))
#         glucose = float(data.get('glucose_level', 0))
#         smoking = data.get('smoking_status', 'Unknown')
#         gender = data.get('gender', 'Unknown')
#         hypertension = data.get('hypertension', '0')
#         heart_disease = data.get('heart_disease', '0')
#         risk_percentage = float(data.get('risk_percentage', 0.0))
#
#         print("Converted age, BMI, glucose, and risk_percentage.")
#         print(f"age = {age}, bmi = {bmi}, glucose = {glucose}")
#
#         # Determine BMI category
#         if bmi < 18.5:
#             bmi_category = "underweight"
#         elif bmi < 25:
#             bmi_category = "normal weight"
#         elif bmi < 30:
#             bmi_category = "overweight"
#         else:
#             bmi_category = "obese"
#
#         # Determine glucose level status
#         if glucose < 100:
#             glucose_status = "normal"
#         elif glucose < 126:
#             glucose_status = "prediabetic range"
#         else:
#             glucose_status = "diabetic range"
#
#         print(f"BMI category = {bmi_category}, Glucose status = {glucose_status}")
#
#         # Create the prompt for OpenAI
#         prompt = f"""As a health advisor, provide personalized health recommendations for a {age}-year-old {gender.lower()} with the following health profile:
#
#         Health Metrics:
#         - BMI: {bmi} ({bmi_category})
#         - Average Glucose Level: {glucose} mg/dL ({glucose_status})
#         - Smoking Status: {smoking}
#         - Hypertension: {"Yes" if hypertension == "1" else "No"}
#         - Heart Disease: {"Yes" if heart_disease == "1" else "No"}
#         - Stroke Risk: {risk_percentage}%
#
#         Please provide:
#         1. An assessment of their current health status
#         2. 3-4 specific, actionable recommendations to improve or maintain their health
#         3. Lifestyle modifications that could help reduce stroke risk
#         4. When to consult healthcare professionals
#
#         Format the response in HTML with proper headings and bullet points. Keep it encouraging and practical.
#         """
#
#         print("Prompt constructed:")
#         print(prompt)
#
#         # Configure OpenAI
#         openai.api_key = "****"
#
#         if not openai.api_key:
#             return JsonResponse({'success': False, 'error': 'OpenAI API key not configured'}, status=500)
#
#         # Call OpenAI with new SDK
#         response = openai.ChatCompletion.create(
#             model="gpt-4",
#             messages=[
#                 {"role": "system",
#                  "content": "You are a helpful health advisor providing personalized, evidence-based health recommendations. Always remind users to consult healthcare professionals for medical decisions."},
#                 {"role": "user", "content": prompt}
#             ],
#             max_tokens=400,
#             temperature=0.4
#         )
#
#         advice = response.choices[0].message['content'].strip()
#         print("Received advice from OpenAI.", advice)
#
#         return JsonResponse({'success': True, 'advice': advice})
#
#     except Exception as e:
#         print("Exception caught in generate_health_advice.", str(e))
#         return JsonResponse({'success': False, 'error': str(e)}, status=500)

import json
import os
from django.http import JsonResponse
from django.shortcuts import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import google.generativeai as genai


@csrf_exempt
@require_http_methods(["POST"])
def generate_health_advice(request):
    print('Generating health advice...')
    try:
        data = json.loads(request.body)
        print("Raw request body parsed.", data)

        # Extract and validate user data
        age = int(float(data.get('age', 0)))
        bmi = float(data.get('bmi', 0))
        glucose = float(data.get('glucose_level', 0))
        smoking = data.get('smoking_status', 'Unknown')
        gender = data.get('gender', 'Unknown')
        hypertension = data.get('hypertension', '0')
        heart_disease = data.get('heart_disease', '0')
        risk_percentage = float(data.get('risk_percentage', 0.0))

        print("Converted age, BMI, glucose, and risk_percentage.")
        print(f"age = {age}, bmi = {bmi}, glucose = {glucose}")

        # Determine BMI category
        if bmi < 18.5:
            bmi_category = "underweight"
        elif bmi < 25:
            bmi_category = "normal weight"
        elif bmi < 30:
            bmi_category = "overweight"
        else:
            bmi_category = "obese"

        # Determine glucose level status
        if glucose < 100:
            glucose_status = "normal"
        elif glucose < 126:
            glucose_status = "prediabetic range"
        else:
            glucose_status = "diabetic range"

        print(f"BMI category = {bmi_category}, Glucose status = {glucose_status}")

        # Create the prompt for Gemini
        prompt = f"""As a health advisor, provide personalized health recommendations for a {age}-year-old {gender.lower()} with the following health profile:

Health Metrics:
- BMI: {bmi} ({bmi_category})
- Average Glucose Level: {glucose} mg/dL ({glucose_status})
- Smoking Status: {smoking}
- Hypertension: {"Yes" if hypertension == "1" else "No"}
- Heart Disease: {"Yes" if heart_disease == "1" else "No"}
- Stroke Risk: {risk_percentage}%

Please provide:
1. An assessment of their current health status
2. 3-4 specific, actionable recommendations to improve or maintain their health
3. Lifestyle modifications that could help reduce stroke risk
4. When to consult healthcare professionals

Format the response in HTML with proper headings and bullet points. Keep it encouraging and practical.
"""

        print("Prompt constructed:")
        print(prompt)

        # Configure Gemini API
        api_key = ""

        genai.configure(api_key=api_key)

        # Initialize the model
        model = genai.GenerativeModel('gemini-1.5-flash')  # Using the cheapest model

        # Generate content
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=400,
                temperature=0.4,
            )
        )

        advice = response.text.strip()
        print("Received advice from Gemini ", advice)

        return JsonResponse({'success': True, 'advice': advice})

    except Exception as e:
        print("Exception caught in generate_health_advice.", str(e))
        return JsonResponse({'success': False, 'error': str(e)}, status=500)
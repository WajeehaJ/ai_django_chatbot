# chat/views.py
from django.contrib import messages
from django.shortcuts import render,  redirect
from .tasks import add_documents_to_VectorDB, run_fine_tuning_LLM
from datetime import datetime
from django.views.decorators.csrf import csrf_exempt

def chat_view(request):
     #To clear all the messages using the function below
    list(messages.get_messages(request))
    str = "chat/chat.html"
    return render(request,str)

def handle_uploaded_file(f): 
    try:
        file_name = 'chat/static/upload/'+ f.name
        with open(file_name, 'wb+') as destination:  
            for chunk in f.chunks():  
                destination.write(chunk)  
            return f.name
        return ""
    except:
        print("Unable to upload the file") 
        return ""
def upload_file(request):
    if request.method == 'POST':
        file_name = handle_uploaded_file(request.FILES['file'])
        if file_name != "":
            messages.success(request, f"Info: File {file_name} uploaded successfully")
        else:
            messages.error(request, f"Error: File {file_name} not uploaded successfully")
        return render(request, 'chat/chat.html', {'file_url': file_name})
    return  render(request, 'chat/chat.html')

def upload_multiple_docs(request):
    if request.method == 'POST':
      
        file_names_list = request.FILES.getlist('multipleFile')
        file_name_output = []
        for file_name in file_names_list:
            output = handle_uploaded_file(file_name)
            if output != "":
                messages.success(request, f"Info: File {output} uploaded successfully")
            else:
                messages.error(request, f"Error: File {output} not uploaded successfully")
            file_name_output.append(output)

        return render(request, "chat/chat.html", {'file_uploaded_url_list': file_names_list})

def add_to_database(request):
    print("Invoking add_documents_to_VectorDB")
    result = add_documents_to_VectorDB.delay()
    return redirect('/')

@csrf_exempt
def trigger_fine_tuning(request):
    if request.method == 'POST':
        file_url = request.POST.get('file_url')
        msg =""
        if file_url:
            result = run_fine_tuning_LLM.delay( "chat/static/upload/"+file_url, "llama2_7b_fine_tune_" + str(datetime.now))
            if result:
                msg = "Fine Tuning Successfully Triggered!!!."
            else: 
                msg = "Error found in fine tuning! Please check your input file again.." 
            return render(request, "chat/chat.html", {'file_url': file_url, 'fine_tuning_result': msg})
    return render(request, "chat/chat.html")
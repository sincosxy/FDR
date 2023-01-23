from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np
import io
import shutil
from datetime import date
import torch
import matplotlib.pyplot as plt
from PIL import Image
import os
from tqdm.notebook import tqdm
from typing import Optional, List
from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import FastAPI, File, Form, UploadFile
from scipy import interpolate, pi

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

resnet = InceptionResnetV1(pretrained='vggface2').eval()

def get_customers(path):
    customers = dict()
    for root, dir, files in os.walk(path):
        for file in files:
            customers[file] = path + file
    return customers

def get_photos(path):
    photos = dict()
    for root, dirs, files in os.walk(path):
        for file in files:
            photos[file] = os.path.join(root, file)
    return photos

def distance(embeddings1, embeddings2, distance_metric=0):
    if distance_metric==0:
        # Euclidian distance
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff),1)
    elif distance_metric==1:
        # Distance based on cosine similarity
        dot = np.sum(np.multiply(embeddings1, embeddings2), axis=1)
        norm = np.linalg.norm(embeddings1, axis=1) * np.linalg.norm(embeddings2, axis=1)
        similarity = dot / norm
        dist = np.arccos(similarity) / pi
    else:
        raise 'Undefined distance metric %d' % distance_metric
    return dist

def get_embeddings(path):
    filebase = []
    embs = dict()
    customers = get_customers(path)
    for customer in customers:
        emb, filepaths = get_embedding(customers[customer])
        filebase.append([customer[:-4], filepaths])
        embs[customer[:-4]] = emb
    return embs, filebase

def predict(filename=None, embeddings=None):
    print('Making prediction...')
    if embeddings == None:
        print("doesn't work yet")
        return False
    if filename == None:
        print("Enter filename")
        return False
        
    embeddings_t, _ = get_embedding(filename)
    distances = []
    name = []
    names = []
    for embedding_t in embeddings_t:
        mindis = 100
        for each in embeddings:
            for emb in embeddings[each]:
                dist = distance(embeddings_t[embedding_t].detach().numpy(), embeddings[each][emb].detach().numpy(), distance_metric=0)
                if dist < mindis:
                    mindis = dist
                    name = [each, dist]
        names.append(name)
    return names

def get_meandis(names):
    unic = list()
    cu = dict()
    sdist = dict()
    for each in names:
        if each[0] not in cu:
            cu[each[0]] = 1
            sdist[each[0]] = each[1]
        else:
            sdist[each[0]] += each[1]
            cu[each[0]] += 1
    for each in cu:
        unic.append([each, (sdist[each][0] / cu[each])])
    return unic

def most_prob(meandists):
    mindis = 100
    for each in meandists:
        if each[1] < mindis:
            mindis = each[1]
            target = each[0]
    return target, mindis

def detect_faces(filename, image_scale = 800, show = False):
    print('Detecting faces...')
    degrees = dict()
    probs = dict()
    dict_faces = dict()
    detector = MTCNN(image_size=160, margin=14, selection_method='largest_over_threshold', keep_all=True) #image_size=160,
    img = Image.open(filename)
    minsize = min(img.size[0], img.size[1])
    if minsize > image_scale:
        img = img.resize((int(img.size[0] * image_scale / minsize), int(img.size[1] * image_scale / minsize)),Image.ANTIALIAS)
    for degree in range(0, 360, 90):
        faces, prob = detector(img.rotate(degree, Image.NEAREST, expand = 1), return_prob=True)
        if faces !=None:
            dict_faces[degree], probs[degree] = faces, np.mean(prob)
    maxprob_idx = max(probs, key=probs.get)
    maxprob_idx = max(probs, key=(lambda k: probs[k]))
    return dict_faces[maxprob_idx], probs

def get_embedding(filename):
    print('Getting embedding...')
    embeddings = dict()
    filepaths = []
    faces, _ = detect_faces(filename)
    i = 0
    for face in faces:
        i += 1
        embedding = resnet(face.unsqueeze(0))
        embeddings[i] = embedding
        filepath = "./faces/" + filename[:-4] + "/"
        os.makedirs(filepath, exist_ok = True)
        torch.save(embedding , os.path.join(filepath,"face" + str(i) + ".pt"))
        filepaths.append(filepath)
    return embeddings, filepaths



def predict_all():
    photolist = get_photos()
    embeddings, _ = get_embeddings()
    errors = 0
    for each in photolist:
    
        label = os.path.basename(os.path.dirname(photolist[each]))
        names = predict(photolist[each], embeddings)
        meandists = get_meandis(names)
        name, prob = most_prob(meandists)
        print(f'Photo: {each}, target: {name}, prob: {prob}, true lbl: {label}')
        if label != name:
            errors += 1
    print(f'Errors: {errors}, acc: {(len(photolist) - errors) / len(photolist)}')
    return {"errors": errors, "acc": (len(photolist) - errors) / len(photolist), "total": len(photolist)}
    


@app.get("/sort")    
async def sort_photos(path: str):
    if path == '':
        today = date.today()
        d3 = today.strftime("%y%m%d")
        path = './groups/' + d3 + "/"
        print(path)
    trainpath, testpath = path + "train/", path + "test/"
    photolist = get_photos(testpath)
    embeddings, _ = get_embeddings(trainpath)
    people = dict()
    for each in photolist:

        names = predict(photolist[each], embeddings)
        meandists = get_meandis(names)
        name, prob = most_prob(meandists)
        person = people.get(name)
        print(f'Photo: {each}, target: {name}, prob: {prob}')
        os.makedirs(testpath + name, exist_ok = True)
        shutil.copyfile(photolist[each], testpath + name + "/" + each)
        if person:
            people[name] += 1
        else:
            people[name] = 1
    print(people)
    return {"count": len(photolist), "people": people}

    
@app.post("/uploadancor")
async def uploadancor(file: bytes = File(), email: str = Form()):
    today = date.today()
    d3 = today.strftime("%y%m%d")
    path = './groups/' + d3 + '/train/'
    os.makedirs(path, exist_ok = True)
    image = Image.open(io.BytesIO(file))
    image.save(path + email + '.jpg')
    image.close()
    return {
        "file_size": len(file),
        "email": email,
        "path": path,
    }


@app.post("/uploadtarget/")
async def uploadtarget(files: List[bytes] = File()):
    today = date.today()
    d3 = today.strftime("%y%m%d")
    path = './groups/' + d3 + '/test/'
    os.makedirs(path, exist_ok = True)
    i=0
    for each in files:
        i += 1
        image = Image.open(io.BytesIO(each))
        image.save(path + str(i) + '.jpg')
        image.close()
    return {"file_sizes": [len(file) for file in files], "count": i, "path": path}


@app.get("/", response_class=HTMLResponse)
async def index_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
    

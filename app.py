from flask import Flask, render_template, request, redirect, url_for, flash, send_file, jsonify, session
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_bcrypt import Bcrypt
from pymongo import MongoClient
from datetime import datetime
import base64
from bson.objectid import ObjectId
from io import BytesIO
import zipfile
from PIL import Image
import io
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import urllib.request
import time
import os
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import threading
import torch
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn
import shutil
from torchvision.models import mobilenet_v3_small
from torchvision.models import mobilenet_v3_large
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY")
bcrypt = Bcrypt(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
MONGO_URI = os.getenv("MONGO_URI")
# MongoDB 설정
client = MongoClient(MONGO_URI)  # MongoDB 서버에 연결
db = client['flask_app']
users_collection = db['users']
login_history_collection = db['login_history']
search_history_collection = db['search_history']
image_collection = db['images']

class User(UserMixin):
    def __init__(self, id, username, email, password_hash, is_admin=False):
        self.id = id
        self.username = username
        self.email = email
        self.password_hash = password_hash
        self.is_admin = is_admin

@login_manager.user_loader
def load_user(user_id):
    user_data = users_collection.find_one({"id": int(user_id)})
    if user_data:
        return User(user_data['id'], user_data['username'], user_data['email'], user_data['password_hash'], user_data['is_admin'])
    return None

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user_data = users_collection.find_one({"email": email})

        if user_data and bcrypt.check_password_hash(user_data['password_hash'], password):
            user = User(user_data['id'], user_data['username'], user_data['email'], user_data['password_hash'], user_data['is_admin'])
            login_user(user)
            
            # 로그인 시간 기록
            login_history_collection.insert_one({
                "user_id": user.id,
                "username": user.username,
                "login_time": datetime.now()
            })
            
            return redirect(url_for('search'))  # 로그인 성공 시 검색 페이지로 리다이렉트
        else:
            flash('Invalid email or password')
            return redirect(url_for('login'))
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if password != confirm_password:
            flash('Passwords do not match')
            return redirect(url_for('signup'))

        password_hash = bcrypt.generate_password_hash(password).decode('utf-8')

        user_id = users_collection.count_documents({}) + 1
        new_user = User(user_id, username, email, password_hash)
        
        users_collection.insert_one({
            "id": user_id,
            "username": username,
            "email": email,
            "password_hash": password_hash,
            "is_admin": False
        })

        flash('Signup successful. Please log in.')
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/index')
@login_required
def index():
    return render_template('home.html')

@app.route('/')
def home():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    return render_template('home.html')

@app.route('/search', methods=['GET', 'POST'])
@login_required
def search():
    if request.method == 'POST':
        query = request.form['query']
        strength=request.form['strong']
        quantity=request.form['quantity']
        strength=float(strength)
        quantity=int(quantity)+1
        print(strength)
        # 검색어 기록
        search_history_collection.insert_one({
            "user_id": current_user.id,
            "username": current_user.username,
            "query": query,
            "search_time": datetime.now()
        })
        driver = webdriver.Chrome() 
        driver.get("https://www.google.co.kr/imghp?hl=ko&tab=wi&authuser=0&ogbl") 
        elem=driver.find_element(By.NAME, 'q')
        time.sleep(2)
        elem.send_keys(query)
        elem.send_keys(Keys.RETURN)
        time.sleep(2)
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        div_tag = soup.find('div', class_='GyAeWb gIatYd')
        imgs = div_tag.find_all('img', class_='YQ4gaf',style=True)
        var=len(imgs)
        count = 1
        while(count<quantity):
            var2=var
            for img in imgs:
                if count>=quantity:
                    break
                try:
                    driver.find_element(By.ID,str(img['id'])).click()
                    time.sleep(2)
                    main=driver.window_handles
                    for i in main:
                        if i!=main[0]:
                            driver.switch_to.window(i)
                            driver.close()
                    driver.switch_to.window(main[0])
                    soup = BeautifulSoup(driver.page_source, 'html.parser')
                    imgUrl=soup.find('img',attrs = {'class':'sFlh5c'})
                    imgUrl=imgUrl["src"] 
                    directory='static/images/'+query
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                    model = YOLO("model/voc_yolon_best.pt")
                    results = model([imgUrl],stream=True)  
                    for result in results:
                        result.save_crop(save_dir=directory)  
                        print('#####',count)
                        count = count + 1
                except:
                        pass
                    # 페이지 맨 아래로 스크롤
            scroll_down(driver)
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            div_tag = soup.find('div', class_='GyAeWb gIatYd')
            imgs = div_tag.find_all('img', class_='YQ4gaf',style=True)
            var=len(imgs)
            imgs=imgs[var2:]
        start_time = time.time()
        path=directory+'/'+query
        percen=len(os.listdir(path))
        print('prev_positive',percen)
        print("Start")

        # 샴넷
        transform = transforms.Compose([
            transforms.Resize((320, 320)),
            transforms.ToTensor(),
        ])
        move='static/images/'+query 
        path=directory+'/'+query
        percen=len(os.listdir(path))
        print('prev_positive',percen)
        keyword=os.listdir(move)
        for l in keyword:
            if l !=query:
                movepath=move+'/'+l
                face=os.listdir(movepath)              
                for m in face:
                    movepath2=movepath+'/'+str(m)
                    face_img=Image.open(movepath2)
                    anchorimage='static/anchors/final/'+query+'.png'
                    anchorImage=Image.open(anchorimage)
                    anchorImage = transform(anchorImage).unsqueeze(0)
                    givenImage = transform(face_img).unsqueeze(0)
                    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                    model=torch.load('model/final/0.pt',map_location=torch.device('cpu'),weights_only=False).to(device)
                    anchorImage=anchorImage.to(device)
                    givenImage=givenImage.to(device)
                    output1, output2 = model(anchorImage,givenImage)
                    distance=F.pairwise_distance(output1,output2)
                    sum=round(float(distance),4)
                    
                    if sum<strength: 
                        print('keyword')                        
                        current_파일명 = movepath2
                        new_파일명 = movepath2[:-3]+l+'.jpg'
                        os.rename(current_파일명, new_파일명)
                        move_파일명=move+'/'+query
                        shutil.move(new_파일명,move_파일명)
            else:
                movepath=move+'/'+l
                face=os.listdir(movepath)
                
                for m in face:
                    
                    movepath2=movepath+'/'+str(m)
                    face_img=Image.open(movepath2)
                    anchorimage='static/anchors/final/'+query+'.png'
                    anchorImage=Image.open(anchorimage)
                    anchorImage = transform(anchorImage).unsqueeze(0)
                    givenImage = transform(face_img).unsqueeze(0)
                    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                    model=torch.load('model/final/0.pt',map_location=torch.device('cpu'),weights_only=False).to(device)
                    anchorImage=anchorImage.to(device)
                    givenImage=givenImage.to(device)
                    output1, output2 = model(anchorImage,givenImage)
                    distance=F.pairwise_distance(output1,output2)
                    sum=round(float(distance),4)

                    if sum>=strength: 
                        print('not keyword')
                        current_파일명 = movepath2
                        new_파일명 = movepath2[:-3]+l+'.jpg'
                        os.rename(current_파일명, new_파일명)
                        move_파일명 = move+'/new'
                        if not os.path.exists(move_파일명):
                             os.makedirs(move_파일명)
                        shutil.move(new_파일명,move_파일명) 
       
        print("End")
        print("Time: {:.4f}sec".format((time.time() - start_time)))
        
        path=directory+'/'+query
        #path=directory
        tage=len(os.listdir(path))
        print('positive',tage)
        images=os.listdir(path)
        image=[0]*len(images)
        for j in range(len(images)):
            image[j]=path+'/'+images[j]
        if "search_query" in session:  # 세션이 있는 경우 세션 삭제
            session.pop('search_query',None)
        if 'search_results' in session:
            session.pop('search_results', None)
        # 페이지네이션을 위해 세션에 이미지 저장
        session['search_results'] = image
        session['search_query'] = query
        return redirect(url_for('search_results', page=1))
    return render_template('search.html')


@app.route('/search_results/<int:page>')
@login_required
def search_results(page):
    per_page = 12
    images = session.get('search_results', [])
    query = session.get('search_query', '')
    total_pages = (len(images) + per_page - 1) // per_page
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    images_paginated = images[start_idx:end_idx]
    return render_template('search_results.html', images=images_paginated, query=query, page=page, total_pages=total_pages)

@app.route('/saved_images', defaults={'page': 1})
@app.route('/saved_images/<int:page>')
@login_required
def saved_images(page):
    per_page = 12
    total_images = image_collection.count_documents({"user_id": current_user.id})
    total_pages = (total_images + per_page - 1) // per_page
    start_idx = (page - 1) * per_page
    images_paginated = image_collection.find({"user_id": current_user.id}).skip(start_idx).limit(per_page)
    
    saved_images = []
    for image in images_paginated:
        image['_id'] = str(image['_id'])  # ObjectId를 문자열로 변환
        saved_images.append(image)
    
    return render_template('saved_images.html', saved_images=saved_images, page=page, total_pages=total_pages)

@app.route('/save_image', methods=['POST'])
@login_required
def save_image():
    image_url = request.form['image_url']
    query = request.form['query']

    #이미지를 중복 확인 후 MongoDB에 저장
    if not image_collection.find_one({"user_id": current_user.id, "image_url": image_url}):
        image_collection.insert_one({
            "user_id": current_user.id,
            "query": query,
            "image_url": image_url,
            "search_time": datetime.now()
        })
    return redirect(url_for('saved_images'))

@app.route('/save_images_bulk', methods=['POST'])
@login_required
def save_images_bulk():
    try:
        image_urls = request.form.getlist('image_urls')
        query = request.form.get('query')

        # 이미지 URL 리스트와 쿼리를 사용하여 필요한 처리 수행
        for image_url in image_urls:
            if not image_collection.find_one({"user_id": current_user.id, "image_url": image_url}):
                image_collection.insert_one({
                    "user_id": current_user.id,
                    "query": query,
                    "image_url": image_url,
                    "save_time": datetime.now()
                })

        flash('Images saved successfully.')
        return redirect(url_for('saved_images'))

    except Exception as e:
        flash(f'Error: {str(e)}')
        return redirect(url_for('search_results', page=1))

@app.route('/delete_image', methods=['POST'])
@login_required
def delete_image():
    image_ids = request.form.getlist('image_ids')
    for image_id in image_ids:
        image_collection.delete_one({"_id": ObjectId(image_id)})
    return redirect(url_for('saved_images'))

@app.route('/download_images', methods=['POST'])
@login_required
def download_images():
    try:
        if request.is_json:
            data = request.get_json()
            filenames = data.get('filenames')
        else:
            filenames = request.form.getlist('filenames')

        zip_filename = 'photos.zip'

        # Zip 파일 생성
        zip_stream = BytesIO()
        with zipfile.ZipFile(zip_stream, 'w', zipfile.ZIP_DEFLATED) as zf:
            for image_id in filenames:
                image_data = image_collection.find_one({"_id": ObjectId(image_id)})
                if image_data:
                        image_path=image_data['image_url']
                        if os.path.exists(image_path):
                            with open(image_path, 'rb') as image_file:
                                    image_name = os.path.basename(image_path)
                                    zf.writestr(image_name, image_file.read())

        zip_stream.seek(0)

        return send_file(
            zip_stream,
            as_attachment=True,
            download_name=zip_filename,
            mimetype='application/zip'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':

    class SiameseNetwork(nn.Module):
        def __init__(self):
            super(SiameseNetwork, self).__init__()
            self.model=mobilenet_v3_small(pretrained=True)
            self.fc = nn.Sequential(
                nn.Linear(576,144),
                nn.ELU(inplace=True),
                nn.Linear(144, 96),
                nn.ELU(inplace=True),
                nn.Linear(96, 64),
            )
            self.model.classifier=self.fc

        def forward(self, input1, input2):
            output1 = self.model(input1)
            output2 = self.model(input2)
            return output1, output2
    siamese_network = SiameseNetwork()

        # 페이지를 아래로 스크롤하는 함수
    def scroll_down(driver):
            # 페이지 맨 아래로 스크롤
            driver.find_element(By.XPATH, '//body').send_keys(Keys.END)
            time.sleep(3)
            try:
                # '더보기' 버튼이 보이면 클릭
                load_more_button = driver.find_element(By.XPATH, '//*[@id="islmp"]/div/div/div/div/div[1]/div[2]/div[2]/input')
                if load_more_button.is_displayed():
                    load_more_button.click()
            except:
                pass
            time.sleep(3)

    app.run()

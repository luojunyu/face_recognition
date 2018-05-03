#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = "$Face Detection API"
__author__ = "$luojunyu"
__mtime__ = "$2018.3.25"
__version__ = "$2.0"
# code is far away from bugs with the god animal protecting
I love animals. They taste delicious.
 ┏┓   ┏┓
┏┛┻━━━┛┻┓
┃       ┃
┃ ┳┛ ┗┳ ┃
┃ ┻   ┻ ┃
┗━┓   ┏━┛
┃ ┗━━━┓
┃神兽保佑   ┣┓
┃永无BUG！ ┏┛
┗┓┓┓ ┏━┳┓┏┛
 ┃┫┫ ┃┫┫
 ┗┻┛ ┗┻┛
"""

import BaseHTTPServer
from BaseHTTPServer import BaseHTTPRequestHandler
import cgi
from SimpleHTTPServer import SimpleHTTPRequestHandler
from SocketServer import ThreadingMixIn
import json
import os
import base64
import uuid
import hkslib
import tools.facerecognition
import cv2
import numpy as np
import time
import datetime
from PIL import Image
import shutil
import gc

save_img_Probability = 10
tolerance = 0.35

detector = hkslib.get_frontal_face_detector()

faces = os.listdir("face/")
faceencodings = []
names = []
for face in faces:
    names.append(face.split(".")[0])
    face_image = tools.facerecognition.load_image_file("face/" + face)
    face_location = tools.facerecognition.face_locations(face_image)
    faceencodings.append(tools.facerecognition.encodings(face_image=face_image, known_face_locations=face_location)[0])


class features:
    def time_cal(func):
        def wrapper(*args, **kwargs):
            start = time.clock()
            result = func(*args, **kwargs)
            end = time.clock()
            print ("%s running time:%s s" % (func.__name__, end - start))
            return result
        return wrapper

    def save_img(self, image, path):
        id_new = uuid.uuid1()
        img_name = str(id_new) + '.jpg'
        img_byte = base64.b64decode(image)
        dir_new = os.path.join(path, img_name)
        if (os.path.exists(dir_new) == False):
            os.mknod(dir_new)
            fp = open(path + '/' + img_name, 'wb')
            fp.write(img_byte)
            fp.close()
            return id_new


    def compare(self,post_values):

        RGB_img = cv2.imdecode(np.fromstring(base64.b64decode(post_values['params']['RGBimage']), np.uint8), cv2.IMREAD_COLOR)

        Gray_img = cv2.imdecode(np.fromstring(base64.b64decode(post_values['params']['IRimage']), np.uint8), cv2.IMREAD_COLOR)

        small_RGBimg = cv2.resize(RGB_img, (0, 0), fx=1, fy=1)
        small_Grayimg = cv2.resize(Gray_img, (0, 0), fx=1, fy=1)
        face_names = []
        RGBface_locations = tools.facerecognition.face_locations(small_RGBimg)
        s = []
        for (top, right, bottom, left) in RGBface_locations:
            top *= 2
            right *= 2
            bottom *= 2
            left *= 2
            s.append(abs((top - bottom) * (right - left)))
        if len(s) == 0:
            return face_names
        if len(s) != 0:
            maxs = max(s)
            # print "存在" + str(len(s)) + "个人脸且面积分别是：" + str(s)
            #print "最大人脸面积是：" + str(max(s))
            for (top, right, bottom, left) in RGBface_locations:
                top *= 2
                right *= 2
                bottom *= 2
                left *= 2
                if (abs((top - bottom) * (right - left))) == maxs:

                    Grayface_locations = tools.facerecognition.face_locations(small_Grayimg)

                    if len(Grayface_locations) != 0:

                        for location in Grayface_locations:

                            if (location[2] - location[0]) > 40:

                                RGBface = small_RGBimg[location[0] - 30:location[2] + 30,
                                          location[3] - 30:location[1] + 30]

                                if (RGBface.size > 0):

                                    rgblocal = tools.facerecognition.face_locations(RGBface)
                                    if len(rgblocal) != 0:

                                        for RGBlocation in rgblocal:

                                            face_encoding = tools.facerecognition.encodings(RGBface,
                                                                                            [RGBlocation])

                                            match = tools.facerecognition.compare(faceencodings,
                                                                                  face_encoding[0],
                                                                                  tolerance=tolerance)

                                            for i in range(len(match)):
                                                if match[i]:
                                                    name = names[i]
                                                    face_names.append(name)
                                            del RGB_img
                                            del Gray_img
                                            del small_RGBimg
                                            del small_Grayimg
                                            del RGBface_locations
                                            del s
                                            del top
                                            del right
                                            del bottom
                                            del left
                                            del maxs
                                            del Grayface_locations
                                            del RGBface
                                            del rgblocal
                                            del face_encoding
                                            del match
                                            del i
                                            del RGBlocation
                                            del post_values
                                            gc.collect()
                                            return face_names

    def add_img(self,image, path):
        try:
            obj_features_add = features()
            id_new=obj_features_add.save_img(image, path)
            img_name = str(id_new) + '.jpg'
            Image.open("temporary/" + img_name)
            face_image = tools.facerecognition.load_image_file("temporary/" + img_name)
            face_location = tools.facerecognition.face_locations(face_image)
            faceencodings.append(
                tools.facerecognition.encodings(face_image=face_image, known_face_locations=face_location)[0])
            shutil.move("temporary/" + img_name, "face")
            return id_new
        except:
            pass


class TodoHandler(BaseHTTPRequestHandler):

    TODOS = []
    def do_GET(self):
        if self.path != '/':
            self.send_error(404, "File not found.")
            return
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write("only post sever")


    def do_POST(self):
        def write_result(str):
            writeresult = file("./log/id_log.txt", 'a+')
            writeresult.write(str + '\n')
            writeresult.close()
            return str
        def send_json(response):
            jresp = json.dumps(response)
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(jresp)

        ctype, pdict = cgi.parse_header(self.headers['content-type'])
        if ctype == 'application/json':
            length = int(self.headers['content-length'])
            post_values = json.loads(self.rfile.read(length))
            rec_service = post_values['service']
            obj_features = features()
            if rec_service == 'compare':
                try:
                    face_names = obj_features.compare(post_values)
                    if len(face_names) == 0:
                        response = {
                            "code": 0,
                            "content":
                                {
                                    "status": False
                                }
                        }
                        send_json(response)
                        del length
                        del post_values
                        # del rec_service
                        del obj_features
                        del response
                        gc.collect()
                        print 'compare image successful, send false'
                    if len(face_names) == 1:
                        face_id = face_names[0]
                        response = {
                            "code": 0,
                            "content":
                                {
                                    "face_id": face_id,
                                    "status": True
                                }
                        }
                        send_json(response)
                        print 'compare image successful, send ture'
                        log_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                        write_result("time:" + log_time + ",  id= " + face_id + " compare successful ")
                        now_time = datetime.datetime.now().strftime('%f')
                        if int(now_time) % save_img_Probability == 0:
                            img_path = os.path.abspath('./' + 'save_img')
                            obj_features.save_img(post_values['params']['RGBimage'], img_path)
                        del log_time
                        del now_time
                        gc.collect()
                    del face_names
                    gc.collect()

                except:
                    response = {
                        "code": 1
                    }
                    send_json(response)
                    print 'compare image failed'

            if rec_service == 'add':
                try:
                    face_path = os.path.abspath('./' + 'face')
                    rec_image = post_values['params']['image']
                    id_new = obj_features.save_img(rec_image, face_path)
                    if id_new != None:
                        response = {
                            "code": 0,
                            "content":
                                {
                                    "face_id": str(id_new)
                                }
                                    }
                        send_json(response)
                        print 'client got code 0 and id'
                except:
                    response = {
                        "code": 1
                    }
                    send_json(response)
                    print 'client got code 1'

            if rec_service == 'delete':
                rec_face_id = post_values['params']['face_id']
                delete_id = rec_face_id + '.jpg'
                face_path = os.path.abspath('./' + 'face')
                files_all = os.listdir(face_path)
                flag = False
                try:
                    for i in range(len(files_all)):
                        if files_all[i] == delete_id:
                            os.unlink(face_path + '/' + delete_id)
                            flag = True
                            break
                    if flag == True:
                        response = {
                            "code": 0,
                        }
                        send_json(response)
                        print 'delete image successful'
                    if flag == False:
                        response = {
                            "code": 1
                        }
                        send_json(response)
                        print 'delete image failed , maybe this image did not exist'
                except:
                    pass

            if rec_service == 'update':
                face_path = os.path.abspath('./' + 'face')
                rec_image = post_values['params']['image']
                rec_face_id = post_values['params']['face_id']
                delete_id = rec_face_id + '.jpg'
                files_all = os.listdir(face_path)
                flag_delete = False
                flag_add = False
                for i in range(len(files_all)):
                    if files_all[i] == delete_id:
                        os.unlink(face_path + '/' + delete_id)
                        flag_delete = True
                        print 'delete successful'
                        break
                if flag_delete:
                    try:
                        img_byte = base64.b64decode(rec_image)
                    except (Exception), e:
                        img_byte = None
                        print e

                    if (img_byte != None):
                        fp = open(face_path + '/' + delete_id, 'wb')
                        fp.write(img_byte)
                        fp.close()
                        flag_add = True
                    print  'add image successful'

                if flag_add and flag_delete == True:
                    response = {
                        "code": 0,
                    }
                    send_json(response)
                    print 'update image successful'

                if flag_delete == False:
                    response = {
                        "code": 1
                    }
                    send_json(response)
                    print 'update image failed , maybe this image did not exist or bad data'

        else:
            self.send_error(415, "Only json data is supported.")
            return

        del rec_service
        gc.collect()
    del TODOS
    gc.collect()


class ThreadingServer(ThreadingMixIn, BaseHTTPServer.HTTPServer):
    pass


def test(HandlerClass=SimpleHTTPRequestHandler,
         ServerClass=BaseHTTPServer.HTTPServer):
    BaseHTTPServer.test(HandlerClass, ServerClass)


if __name__ == '__main__':
    server = ThreadingServer(('192.168.0.58', 8081), TodoHandler)
    print("Starting server, use <Ctrl+C> to stop")
    server.serve_forever()

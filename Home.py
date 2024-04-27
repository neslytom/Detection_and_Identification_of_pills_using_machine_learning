from flask import Flask, render_template, request,flash,session
import mysql.connector
from werkzeug.utils import secure_filename
import os
import io
import base64
from detection import img_prediction
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = "1234"

@app.route('/')
def index():
    return render_template('index.html')

#Admin Section
@app.route('/ahome')
def ahome():
    return render_template('admin_home.html')

@app.route("/admin")
def admin():
    return render_template("admin.html")

@app.route("/admin_login_check",methods =["GET", "POST"])
def admin_login_check():
    uid = request.form.get("unm")
    pwd = request.form.get("pwd")
    if uid=="admin" and pwd=="admin":
        return render_template("admin_home.html")
    else:
        return render_template("admin.html",msg="Invalid Credentials")

@app.route("/model_evaluations")
def model_evaluations():

    return render_template("evaluations.html")

#User Section

@app.route("/user")
def user():
    return render_template("user.html")

@app.route("/user_reg")
def user_reg():
    return render_template("user_reg.html")

@app.route("/user_reg_store",methods =["GET", "POST"])
def user_reg_store():
    name = request.form.get('name')
    uid = request.form.get('uid')
    pwd = request.form.get('pwd')
    email = request.form.get('email')
    mno = request.form.get('mno')
    con, cur = database()
    sql = "select count(*) from users where userid='" + uid + "'"
    cur.execute(sql)
    res = cur.fetchone()[0]
    if res > 0:
        return render_template("user_reg.html", messages="User Id already exists..!")
    else:
        sql = "insert into users values(%s,%s,%s,%s,%s)"
        values = (name, uid, pwd, email, mno)
        cur.execute(sql,values)
        con.commit()
        return render_template("user.html", messages="Registered Successfully..! Login Here.")
    return ""

@app.route("/userlogin_check",methods =["GET", "POST"])
def userlogin_check():
        uid = request.form.get("unm")
        pwd = request.form.get("pwd")
        con, cur = database()
        sql = "select count(*) from users where userid='" + uid + "' and password='" + pwd + "'"
        cur.execute(sql)
        res = cur.fetchone()[0]
        if res > 0:
            session['uid'] = uid
            qry = "select * from users where userid= '" + uid + " ' "
            cur.execute(qry)
            val = cur.fetchall()
            for values in val:
                name = values[0]
                print(name)

            return render_template("user_home.html",name=name)
        else:

            return render_template("user.html",msg="Invalid Credentials")
        return ""

@app.route("/uhome")
def uhome():
    con,cur=database()
    uid = session['uid']
    qry = "select * from users where userid= '" + uid + " ' "
    cur.execute(qry)
    vals = cur.fetchall()
    for values in vals:
        name = values[0]
        print(name)

    return render_template("user_home.html",name=name)

@app.route('/prediction')
def prediction():
    return render_template('detection.html')

@app.route("/detection",methods =["GET", "POST"])
def detection():
    image = request.files['file']
    print("image",image)
    imgdata = secure_filename(image.filename)
    print("imgdata",imgdata)
    filename = image.filename
    print("filename",filename)

    filelist = [f for f in os.listdir("testimg")]
    for f in filelist:
        os.remove(os.path.join("testimg", f))

    image.save(os.path.join("testimg", imgdata))

    image_path = "../PillsDetection/testimg/" + filename

    result = img_prediction(image_path)

    con,cur=database()
    qry="select * from pill_metadata where pill_name= '"+result+"' "
    cur.execute(qry)
    vals = cur.fetchall()
    print(vals)

    return render_template("result.html",result=vals)


#DATABASE CONNECTION
def database():
    con = mysql.connector.connect(host="127.0.0.1", user='root', password="root", database="pills_detection")
    cur = con.cursor()
    return con, cur


if __name__ == '__main__':
    app.run(host="localhost", port=2024, debug=True)

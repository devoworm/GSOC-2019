from flask import Flask, request, render_template, Response
import cv2
import numpy as np
import base64
import pandas as pd

app = Flask(__name__)

df = pd.DataFrame(columns=['name', 'cell', 'X coordinate', 'Y coordinate', 'area'])


@app.route("/", methods = ["GET","POST"])
def home():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file found"
        final_list = {}
        c = 0
        row = 0
        df.drop(df.index, inplace=True)
        for i in request.files.getlist("file"):
            print(i, end='\n')
            print(i.filename)
            file_name_str = str(i.filename)
            file_name_str = file_name_str.split('.')[0]
            img = i.read()

            img = cv2.imdecode(np.frombuffer(img, np.uint8), -1)
            img = cv2.resize(img, (390, 620))
            # blur = cv2.medianBlur(img, 11) # median blurring
            blur = cv2.bilateralFilter(img,9,75,75) # bilateralFilter blurring (seems to have less noise some times)

            # Gradient Morphology
            kernel = np.ones((3,3),np.uint8)
            gradient = cv2.morphologyEx(blur, cv2.MORPH_GRADIENT, kernel)
            op = cv2.morphologyEx(gradient, cv2.MORPH_OPEN, kernel)
            gray = cv2.cvtColor(op, cv2.COLOR_BGR2GRAY)

            ret,thresh1 = cv2.threshold(gray,1,255,cv2.THRESH_BINARY)

            conts, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            filter_c = []
            for i in range(len(conts)):
                area = cv2.contourArea(conts[i])
                if area>500:
                    if (hierarchy[0][i][3] != -1):
                        filter_c.append(conts[i])
                        # filter_h.append(hierarchy[0][i])

            for j in range(len(filter_c)):
                # compute the center of the contour
                M = cv2.moments(filter_c[j])
                area = cv2.contourArea(filter_c[j])
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                df.loc[row] = [file_name_str] + ['cell'+str(j+1)] + [cX] + [cY] + [area]
                row+=1
                cv2.drawContours(img, [filter_c[j]], -1, (0, 255, 0), 4)
                cv2.circle(img, (cX, cY), 4, (255, 255, 255), -1)

            ret, buf = cv2.imencode( '.jpg', img )

            img_string = base64.b64encode(buf)
            img_string = img_string.decode("utf-8")

            final_list[str(c)] = img_string
            c+=1

        return render_template("result.html", final_list = final_list)
    else:
        return render_template("index.html")



@app.route('/downloadcsv' )
def downloadcsv():
    csv = df.to_csv(index=False)
    return Response(
        csv,
        mimetype="text/csv",
        headers={"Content-disposition":
                 "attachment; filename=SPIM_images_analysis.csv"})

if __name__ == "__main__":
    app.run()
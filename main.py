import os
import numpy as np
import pandas as pd
from skimage import io, transform
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from flask import Flask, request, render_template

category = ["apple", "banana"]
datadir = "dataset"

data = []
target = []

for i in category:
  path = os.path.join(datadir, i)
  print(f"Loading {i} data...")
  iter = 0
  for j in os.listdir(path):
    iter += 1
    print(f"Loaded {iter}/{len(os.listdir(path))}")
    img = io.imread(os.path.join(path, j), as_gray=True)
    img = transform.resize(img, (64, 64))
    hog_features = hog(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys')
    data += [hog_features]
    target += [category.index(i)]

data = np.array(data)
target = np.array(target)

df = pd.DataFrame(data)
df["target"] = target

X=df.iloc[:,:-1]
y=df.iloc[:,-1]

print("Splitting the data...")
X_train, X_test, y_train, y_test = train_test_split(X, y)

param_grid={
    'C':[0.1,1,10,100],
    'gamma':[0.0001,0.001,0.1,1],
    'kernel':['rbf','poly']
}

print("Training the model...")
svc = SVC(probability=True)
clf = GridSearchCV(svc,param_grid)
clf.fit(X_train, y_train)

print("Testing model accuracy...")
predictions = clf.predict(X_test)
accuracy = accuracy_score(predictions, y_test)
print(f"Accuracy: {accuracy}")

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
  context = {}
  if request.method == "POST":
    file = request.files["image"]
    file.save(file.filename)

    img = io.imread(file.filename, as_gray=True)
    os.remove(file.filename)
    img = transform.resize(img, (64, 64))
    hog_features = hog(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys')
    context["prediction"] = category[clf.predict([hog_features])[0]]
  return render_template("index.html", context=context)

if __name__ == "__main__":
  print("Starting server")
  host = os.getenv("IP", "0.0.0.0")
  port = int(os.getenv("PORT", 5000))
  app.run(host=host, port=port)
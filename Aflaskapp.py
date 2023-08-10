from flask import Flask, render_template
import os

app = Flask(__name__)
    
    
@app.route('/')
def mainPage():
    #path = os.path.abspath('mainPage.html')
    try:
        return render_template('mainPage.html')
    except Exception as e:
        return str(e)
    #return 'text on mainPage'
    
    
@app.route("/classifierResults/")
def classifierResults():
    try:
        return render_template("classifierResults.html")
    except Exception as e:
        return str(e)


@app.route("/objDetResults/")
def objDetResults():
    try:
        return render_template("objDetResults.html")
    except Exception as e:
            return str(e)


@app.route("/allRawData/")
def allRawData():
    try:
        return render_template("allRawData.html")
    except Exception as e:
            return str(e)


@app.route("/processedVideos/")
def processedVideos():
    try:
        return render_template("processedVideos.html")
    except Exception as e:
            return str(e)


@app.route("/unprocessedVideos/")
def unprocessedVideos():
    try:
        return render_template("unprocessedVideos.html")
    except Exception as e:
            return str(e)
    
        
@app.route("/processedFrames/")
def processedFrames():
    try:
        return render_template("processedFrames.html")
    except Exception as e:
            return str(e)


if __name__ == '__main__':
    app.run()
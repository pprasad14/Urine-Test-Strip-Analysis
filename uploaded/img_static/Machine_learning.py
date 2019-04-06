from flask import Flask, render_template, request, jsonify
import os

app = Flask(__name__)

# UPLOAD_FOLDER = r'static'
UPLOAD_FOLDER = r'C:\x\Docs\python\Cumulations_practice\python-virtual-environments\Vir\static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route("/")
def show():
	return render_template('Machine_learning.html');
	# pass

@app.route('/upload', methods = ['POST'])
def upload():
	try:
		image = request.files['img']
		
		image.save(os.path.join(app.config['UPLOAD_FOLDER'],image.filename))

		d = {	"GLU":"Neg",
				"PH":"5.0",
				"BIL":"Neg",
				"NIT":"Positive",
				"KET":"Neg",
				"BLO":"Neg",
				"URO":1,
				"SG":">=1.030",
				"LEU":"Neg",
				"PRO":"Trace"
			}
		return jsonify(d)

	except:
		# return jsonify({'error':'upload error'})
		return ({'error':'upload error'})
		
if __name__ == "__main__":
    app.run()
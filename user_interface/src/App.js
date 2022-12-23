import './App.css';

function App() {

    let path = ""

    const getPath = (file) => {
        path = file.path.split(file.name)[0]
    }

    const runScript = () => {
        var { PythonShell } = require('python-shell');
        let obj = {
            "brisque": document.getElementById("brisque").checked,
            "nima": document.getElementById("nima").checked,
            "labels": document.getElementById("labels").checked,
            "objects": document.getElementById("objects").checked,
            "slideshow": document.getElementById("slideshow").checked,
            "path": path
        }

        let options = {
            mode: 'text',
            args: [`-p ${obj.path}`, `-a ${obj.brisque ? ' b' : ''}${obj.nima ? ' n' : ''}${obj.labels ? ' l' : ''}${obj.objects ? ' o' : ''}${obj.slideshow ? ' s' : ''}`]
        };

        PythonShell.run('../main.py', options, function (err, results) {
            if (err) throw err;
            // results is an array consisting of messages collected during execution
            document.getElementById('output').value = document.getElementById('output').value + results[0]
            console.log('results: ', results);
        });
    }

    return (
        <div className="center">
            <div className="form-group">
                <h4 className="title">Configuration:</h4>
                <div className="form-check form-switch">
                    <input className="form-check-input" type="checkbox" id="brisque" defaultChecked />
                    <label className="form-check-label" htmlFor="brisque">Tecnical evaluation (BRISQUE)</label>
                </div>
                <div className="form-check form-switch">
                    <input className="form-check-input" type="checkbox" id="nima" defaultChecked />
                    <label className="form-check-label" htmlFor="nima">Aesthetics evaluation (NIMA)</label>
                </div>
                <div className="form-check form-switch">
                    <input className="form-check-input" type="checkbox" id="labels" defaultChecked />
                    <label className="form-check-label" htmlFor="labels">Label images (Used for image clustering)</label>
                </div>
                <div className="form-check form-switch">
                    <input className="form-check-input" type="checkbox" id="objects" defaultChecked />
                    <label className="form-check-label" htmlFor="objects">Identify objetcs in images (Used for image clustering)</label>
                </div>
                <div className="form-check form-switch">
                    <input className="form-check-input" type="checkbox" id="slideshow" defaultChecked />
                    <label className="form-check-label" htmlFor="slideshow">Create slideshow</label>
                </div>
                <div className="form-group">
                    <label htmlFor="exampleFormControlFile1">Choose your photos folder: </label>
                    <input directory="" webkitdirectory="" type="file" onChange={(event) => getPath(event.target.files[0])} className="form-control-file" id="path" />
                </div>
                <center>
                    <div className="button padding">
                        <button className="btn btn-primary" onClick={runScript}>Run</button>
                    </div>
                </center>
                <div className="form-group">
                    <label htmlFor="exampleFormControlTextarea1">Output:</label>
                    <textarea className="form-control rounded-0" id="output" rows="15"></textarea>
                </div>
            </div>
        </div>
    );
}

export default App;

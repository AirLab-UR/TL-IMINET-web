var express = require('express');
var multer  = require('multer');
const { spawn } = require('child_process')

/*
app.get('/foo', function(req, res) {
    // Call your python script here.
    // I prefer using spawn from the child process module instead of the Python shell
    const scriptPath = 'hello.py'
    const process = spawn('python', [scriptPath, arg1, arg2])
    process.stdout.on('data', (myData) => {
        // Do whatever you want with the returned data.
        // ...
        res.send("Done!")
    })
    process.stderr.on('data', (myErr) => {
        // If anything gets written to stderr, it'll be in the myErr variable
    })
})
*/


var app = express();

app.use(express.static('public')); // for serving the HTML file

var upload = multer({ dest: __dirname + '/public/uploads/' });
var type = upload.single('audio_data');


app.post('/api/test', type, function (req, res, next) {
   console.log(req.body);
   console.log(req.file);
   // do stuff with file
});

app.listen(8081);

app.get('/name', callName);

function callName(req, res) {
    var spawn = require("child_process").spawn;
    var process = spawn('python', ["./hello.py"]);
    process.stdout.on('data', function(data){
        res.send(data.toString());
    })
}
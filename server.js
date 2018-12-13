/*
var express = require('express');
var multer = require('multer');
var app = express();

app.use(express.static('public'));

//app.get('/index.html', function (req, res) {
//   res.sendFile( __dirname + "/" + "index.html" );
//})

var upload = multer({ dest: __dirname + '/public/uploads/' });
var type = upload.single('upl');

app.post('/api/test', type, function (req, res) {
   console.log(req.body);
   console.log(req.file);
   // do stuff with file
});

var server = app.listen(8081, function () {
   var host = server.address().address
   var port = server.address().port

   console.log("Example app listening at http://%s:%s", host, port)
})
*/

var express = require('express');
var fs = require('fs');
var https = require('https');

var options = {
    key: fs.readFileSync('/etc/apache2/ssl/apache.key'),
    cert: fs.readFileSync('/etc/apache2/ssl/apache.crt'),
    requestCert: false,
    rejectUnauthorized: false
};

var multer  = require('multer');
var app = express();
//var server = require('http').createServer(app); 
var server = https.createServer(options, app);

var io = require('socket.io')(server);

var clickCount = 0;

//app.use(express.static('public')); // for serving the HTML file
app.use(express.static('./')); // for serving the HTML file

var storage = multer.diskStorage({
	destination: function (req, file, cb) {
		cb(null, __dirname + '/public/uploads')
	},
	filename: function (req, file, cb) {
		cb(null, 'demo.wav')
	}
})
var upload = multer({storage: storage})
//var upload = multer({ dest: __dirname + '/public/uploads/' });

var type = upload.single('audio_data');


app.post('/api/test', type, function (req, res, next) {
   console.log(req.body);
   console.log(req.file);
   // do stuff with file
});

//app.listen(8081);
//console.log('Server running on port 8081');

//var spawn = require("child_process").spawn;
//var process = spawn('python', ["./hello-yichi.py"]);
console.log('start prediction');

io.on('connection', function(client) { 
  console.log('Client connected...'); 
  //when the server receives clicked message, do this
    client.on('clicked', function(data) {
      clickCount++;
      
      var messageString = '';

      //var sys = require('sys');
      var spawn = require("child_process").spawn;
      var process = spawn('python', ["./hello-yichi.py"]);
      process.stdout.on('data', function(data) {
      //console.log('stdout: ${data}');
      //res.send(data.toString());
      messageString += data.toString();
      });
      console.log('python data retrieved');
      
      process.stdout.on('end', function(){
      //console.log('Data=',messageString);
      //send a message to ALL connected clients
      io.emit('buttonUpdate', messageString);
      });

    });
});


server.listen(8081, function(){
  console.log('listening on *:8081');
}); 
/*
app.get('/name', callName);

function callName(req, res) {
    var spawn = require("child_process").spawn;
    var process = spawn('python', ["./hello.py"]);
    
    process.stdout.on('data', (data) =>{
    	//console.log('stdout: ${data}');
    	res.send(data.toString());
    	console.log(data.toString());
    })
    
    //var output = "";
    //process.stdout.on('data', function(data){
    //    output += data;
    //    console.log(data);
    //    //res.send(data.toString());
    //
    
    console.log('Finished calling python script');
}
*/
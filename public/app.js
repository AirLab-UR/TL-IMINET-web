//webkitURL is deprecated but nevertheless
URL = window.URL || window.webkitURL;

var gumStream; 						//stream from getUserMedia()
var rec; 							//Recorder.js object
var input; 							//MediaStreamAudioSourceNode we'll be recording

// shim for AudioContext when it's not avb. 
var AudioContext = window.AudioContext || window.webkitAudioContext;
var audioContext //audio context to help us record

var recordButton = document.getElementById("recordButton");
var stopButton = document.getElementById("stopButton");
var pauseButton = document.getElementById("pauseButton");
var predictButton = document.getElementById("predictButton");

//add events to those 3 buttons
recordButton.addEventListener("click", startRecording);
stopButton.addEventListener("click", stopRecording);
pauseButton.addEventListener("click", pauseRecording);
//predictButton.addEventListener("click", predictRecording);


function startRecording() {
	console.log("recordButton clicked");

	/*
		Simple constraints object, for more advanced audio features see
		https://addpipe.com/blog/audio-constraints-getusermedia/
	*/
    
    document.getElementById("buttonAppear").innerHTML = '';
    document.getElementById("buttonPredict").innerHTML = '';  // Newly added by Yichi
    document.getElementById("predictTitle").innerHTML = '';
    document.getElementById("buttonCount").innerHTML = '';
    document.getElementById("Recording").innerHTML = '';
    document.getElementById("recordingsList").innerHTML = '';
    document.getElementById("waveform").style.display = "none";

	document.getElementById("ProgressBar").innerHTML = 'Recording...';

    var constraints = { audio: true, video:false }

 	/*
    	Disable the record button until we get a success or fail from getUserMedia() 
	*/

	recordButton.disabled = true;
	stopButton.disabled = false;
	pauseButton.disabled = false;
	//predictButton.disabled = true

	/*
    	We're using the standard promise based getUserMedia() 
    	https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia
	*/

	navigator.mediaDevices.getUserMedia(constraints).then(function(stream) {
		console.log("getUserMedia() success, stream created, initializing Recorder.js ...");

		/*
			create an audio context after getUserMedia is called
			sampleRate might change after getUserMedia is called, like it does on macOS when recording through AirPods
			the sampleRate defaults to the one set in your OS for your playback device

		*/
		audioContext = new AudioContext();

		//update the format 
		//document.getElementById("formats").innerHTML="Format: 1 channel pcm @ "+audioContext.sampleRate/1000+"kHz"

		/*  assign to gumStream for later use  */
		gumStream = stream;
		
		/* use the stream */
		input = audioContext.createMediaStreamSource(stream);

		/* 
			Create the Recorder object and configure to record mono sound (1 channel)
			Recording 2 channels  will double the file size
		*/
		rec = new Recorder(input,{numChannels:1})

		//start the recording process
		rec.record()

		console.log("Recording started");

	}).catch(function(err) {
	  	//enable the record button if getUserMedia() fails
    	recordButton.disabled = false;
    	stopButton.disabled = true;
    	pauseButton.disabled = true
	});
}

function pauseRecording(){
	console.log("pauseButton clicked rec.recording=",rec.recording );
	if (rec.recording){
		//pause
		rec.stop();
		pauseButton.innerHTML="Resume";
	}else{
		//resume
		rec.record()
		pauseButton.innerHTML="Pause";

	}
}

function stopRecording() {
	document.getElementById("waveform").style.display = "block";
	console.log("stopButton clicked");
	document.getElementById("ProgressBar").innerHTML = '';
	document.getElementById("Recording").innerHTML = '<h3>Recording</h3>';


	//disable the stop button, enable the record too allow for new recordings
	stopButton.disabled = true;
	recordButton.disabled = false;
	pauseButton.disabled = true;
	//predictButton.disabled = false;

	//reset button just in case the recording is stopped while paused
	pauseButton.innerHTML="Pause";
	
	//tell the recorder to stop the recording
	rec.stop();

	//stop microphone access
	gumStream.getAudioTracks()[0].stop();

	//create the wav blob and pass it on to createDownloadLink
	rec.exportWAV(createDownloadLink);

	recordButton.innerHTML="Rerecord";
}

function createDownloadLink(blob) {
	//predictButton.disabled = false;

	var url = URL.createObjectURL(blob);

    /*
	var wavesurfer = WaveSurfer.create({
     container: '#waveform',
     waveColor: 'grey',
      progressColor: 'hsla(200, 100%, 30%, 0.5)',
      cursorColor: '#333',
      //barWidth: 3
    });*/

    wavesurfer.load(url);
    wavesurfer.on('ready', function () {
        //console.log('Success');
        //wavesurfer.play();
        var playpause = document.getElementById("buttonAppear").innerHTML = 
        '<button onclick="wavesurfer.playPause()">Play / Pause</button>'+
        '<button onclick="loadModel()">Load</button>'+
        '<button onclick="predictRecording()">Predict</button>';
        //var predict = document.getElementById("buttonPredict").innerHTML = 
        //'<button class="btn btn-primary" onclick="myFunction()">Predict</button>';  // Newly added by Yichi
        
    });

	var li = document.getElementById('recordingsList');
	document.getElementById('recordingsList').innerHTML = ''; //clear previous name and link
	var link = document.createElement('a');
	var au = document.createElement('audio');

	//name of .wav file to use during upload and download (without extendion)
	var filename = new Date().toISOString();

	//add controls to the <audio> element
	//au.controls = false;
	au.src = url;

	//save to disk link
	link.href = url;
	link.download = filename+".wav"; //download forces the browser to donwload the file using the  filename
	link.innerHTML = "Save";



	//add the new audio element to li
	//li.appendChild(au);
	
	//add the filename to the li
	li.appendChild(document.createTextNode(filename+".wav "))
	//li.appendChild(document.createTextNode(au.duration))

	//add the save to disk link to li
	li.appendChild(link);
	
	//upload link
	var upload = document.createElement('a');
	//upload = document.getElementById("buttonPredict");
	upload.href="#";
	upload.innerHTML = "Upload";
	upload.addEventListener("click", function(event){
		  //socket.emit('clicked');//added by yiting
		  var xhr=new XMLHttpRequest();
		  xhr.onload=function(e) {
		      if(this.readyState === 4) {
		          console.log("Server returned: ",e.target.responseText);
		      }
		  };
		  var fd = new FormData();
		  fd.append('audio_data', blob, 'demo.wav');
		  // fd.append("audio_data", blob, 'demo.wav');
		  //xhr.open("POST","upload.php",true);
		  //xhr.send(fd);

		  fetch('../api/test',
          {
              method: 'post',
              body: fd
          });
	})
	li.appendChild(document.createTextNode (" "))//add a space in between
	li.appendChild(upload)//add the upload link to li

	au.addEventListener('loadedmetadata', function() {
    //console.log("Playing " + au.src + ", for: " + au.duration.toFixed(2) + "seconds.");
    var br = document.createElement("br");
    li.appendChild(br)//add a space in between

    li.appendChild(document.createTextNode("Recording Length: " + au.duration.toFixed(2) + " seconds"))
    li.appendChild(document.createTextNode (" "))//add a space in between
    });

	//add the li element to the ol
	//recordingsList.appendChild(li);
}

/*
function predictRecording() {
    console.log("predictButton clicked");
    fetch('/name').then(function(response){
    	//response.text().then(function(text){
    	//	posi-pair-gt.textContent = text;
    	//});
    	
    })
    .then(function(myJson){
    	console.log(JSON.stringify(myJson));
    })
}
*/
/*
var socket0 = io.connect();
function loadModel() {
	socket0.emit('clicked');
}
socket0.on('buttonUpdate', function(data){
	document.getElementById("predictTitle").innerHTML = 'Model Loaded.';
});
*/

var socket = io.connect();
function predictRecording() {
	socket.emit('clicked');
}
socket.on('buttonUpdate', function(data){
	document.getElementById("predictTitle").innerHTML = 'Prediction:';
	document.getElementById("buttonCount").innerHTML = 'The returned sounds are: ' + data;
});



let mediaRecorder;
let recordedChunks = [];

async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        document.getElementById('camera').srcObject = stream;

        recordedChunks = [];
        mediaRecorder = new MediaRecorder(stream);
        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                recordedChunks.push(event.data);
            }
        };

        mediaRecorder.onstop = async () => {
    try {
        const blob = new Blob(recordedChunks, { type: 'video/mp4' });
        const form = document.getElementById('uploadForm');
        const language = form.querySelector('#language').value;

        const formData = new FormData();
        formData.append('file', blob, 'recorded_video.mp4');
        formData.append('file_type', 'video');
        formData.append('language', language);

        // Upload video to server
        const response = await fetch('/upload', { method: 'POST', body: formData });

        if (response.ok) {
            const data = await response.json(); // Parse the JSON response
            if (data.redirect_url) {
                // Redirect to the new page
                window.location.href = data.redirect_url;
            } else {
                alert('Processing completed, but no redirect URL provided.');
            }
        } else {
            // Handle server error response
            const errorData = await response.json();
            alert(`Error: ${errorData.error || 'Unknown error occurred.'}`);
        }
    } catch (error) {
        console.error('Error uploading video:', error);
        alert('An unexpected error occurred.');
    }
};


        mediaRecorder.start();
        document.getElementById('startButton').disabled = true;
        document.getElementById('stopButton').disabled = false;
    } catch (error) {
        console.error('Error accessing camera:', error);
    }
}

function stopRecording() {
    mediaRecorder.stop();
    document.getElementById('camera').srcObject.getTracks().forEach((track) => track.stop());
    document.getElementById('startButton').disabled = false;
    document.getElementById('stopButton').disabled = true;
}

function toggleInterface() {
    const fileUploadSection = document.getElementById('fileUploadSection');
    const recordInterface = document.getElementById('recordInterface');
    const selectedFileType = document.querySelector('input[name="file_type"]:checked').value;

    if (selectedFileType === 'Record') {
        // Show recording interface and hide file upload
        recordInterface.style.display = 'block';
        fileUploadSection.style.display = 'none';
    } else {
        // Show file upload and hide recording interface
        recordInterface.style.display = 'none';
        fileUploadSection.style.display = 'block';
    }
}


document.getElementById('uploadButton').addEventListener('click', async () => {
    const form = document.getElementById('uploadForm');
    const formData = new FormData(form);

    console.log(formData)

    try {
        // Send the form data using fetch
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData,
        });

        if (response.ok) {
            const data = await response.json(); // Parse the JSON response
            if (data.redirect_url) {
                // Redirect to the new page
                window.location.href = data.redirect_url;
            } else {
                alert('Processing completed, but no redirect URL provided.');
            }
        } else {
            const errorData = await response.json();
            alert(`Error: ${errorData.error}`);
        }
    } catch (error) {
        console.error('Error uploading file:', error);
        alert('An unexpected error occurred.');
    }
});

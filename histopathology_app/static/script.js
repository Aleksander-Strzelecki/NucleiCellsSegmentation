const form = document.getElementById('uploadForm');
const modal = document.getElementById('resultModal');
const closeModalBtn = document.querySelector('.close');
const resultImage = document.getElementById('resultImage');
const showPredictionCheckbox = document.getElementById('showPrediction');
const loadingSpinner = document.getElementById('loadingSpinner');

let originalImageUrl = '';
let predictionImageUrl = '';

window.onload = function() {
    loadingSpinner.style.display = 'none';
};

closeModalBtn.onclick = function() {
    modal.style.display = 'none';
};

window.onclick = function(event) {
    if (event.target == modal) {
        modal.style.display = 'none';
    }
};

function addTimestamp(url) {
    return url + '?t=' + new Date().getTime();
}

showPredictionCheckbox.onchange = function() {
    if (showPredictionCheckbox.checked) {
        resultImage.src = addTimestamp(predictionImageUrl);
    } else {
        resultImage.src = addTimestamp(originalImageUrl);
    }
};

form.onsubmit = async function(event) {
    event.preventDefault();

    loadingSpinner.style.display = 'flex';

    const formData = new FormData(form);
    const response = await fetch('/predict', {
        method: 'POST',
        body: formData
    });

    if (response.ok) {
        const blob = await response.blob();
        const imgUrl = URL.createObjectURL(blob);
        originalImageUrl = '/models/tmp/original_image.png';
        predictionImageUrl = '/models/tmp/output_image.png';
        
        loadingSpinner.style.display = 'none';
        showPredictionCheckbox.checked = true;

        resultImage.src = addTimestamp(predictionImageUrl);
        modal.style.display = 'block';
    } else {
        alert('Error occurred while predicting.');
        loadingSpinner.style.display = 'none';
    }
};

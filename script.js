const API_BASE = '';
let accuracyChart = null;
let distChart = null;

// UI Navigation
function showSection(id) {
    // Only hide top-level sections inside the container
    const container = document.querySelector('.container');
    Array.from(container.children).forEach(s => s.classList.add('hidden'));

    const target = document.getElementById(id);
    target.classList.remove('hidden');

    if (id === 'dashboard-section') {
        target.style.display = 'grid';
    } else {
        target.style.display = 'block';
    }
}

// Authentication: Signup
document.getElementById('signup-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const username = document.getElementById('signup-username').value;
    const password = document.getElementById('signup-password').value;
    const btn = e.target.querySelector('.btn');
    const originalContent = btn.innerHTML;

    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
    btn.disabled = true;

    try {
        const response = await fetch(`${API_BASE}/signup`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username, password })
        });
        const data = await response.json();
        if (response.ok) {
            alert("Signup successful! Please login.");
            showSection('login-section');
        } else {
            alert(data.detail || "Signup failed");
        }
    } catch (err) {
        alert("Server error. Make sure backend is running.");
    } finally {
        btn.innerHTML = originalContent;
        btn.disabled = false;
    }
});

// Authentication: Login
document.getElementById('login-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const username = document.getElementById('login-username').value;
    const password = document.getElementById('login-password').value;
    const btn = e.target.querySelector('.btn');
    const originalContent = btn.innerHTML;

    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Authenticating...';
    btn.disabled = true;

    try {
        const response = await fetch(`${API_BASE}/login`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username, password })
        });
        const data = await response.json();
        if (response.ok) {
            showSection('dashboard-section');
            initChart();
        } else {
            alert(data.detail || "Login failed");
        }
    } catch (err) {
        alert("Server error. Make sure backend is running.");
    } finally {
        btn.innerHTML = originalContent;
        btn.disabled = false;
    }
});

// Logout
function logout() {
    showSection('login-section');
}

// Sentiment Analysis
async function analyzeSentiment() {
    const text = document.getElementById('text-input').value;
    if (!text) return alert("Please enter some text");

    const btn = document.querySelector('.analysis-section .btn');
    const originalContent = btn.innerHTML;
    btn.innerHTML = '<i class="fas fa-wand-magic-sparkles fa-spin"></i> Analyzing...';
    btn.disabled = true;

    try {
        const response = await fetch(`${API_BASE}/analyze`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text })
        });
        const data = await response.json();

        if (response.ok) {
            updateResult('nb-result', data.results.nb);
            updateResult('svm-result', data.results.svm);
            updateChart(data.metrics);
            document.getElementById('feedback-controls').classList.remove('hidden');
        } else {
            alert(data.detail || "Analysis failed");
        }
    } catch (err) {
        alert("Server error");
    } finally {
        btn.innerHTML = originalContent;
        btn.disabled = false;
    }
}

async function submitFeedback(correctSentiment) {
    const text = document.getElementById('text-input').value;
    const feedbackCard = document.getElementById('feedback-controls');
    const buttons = feedbackCard.querySelectorAll('button');

    buttons.forEach(b => b.disabled = true);

    try {
        const response = await fetch(`${API_BASE}/feedback`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text, sentiment: correctSentiment })
        });
        const data = await response.json();

        if (response.ok) {
            alert("Model Updated! It has learned your correction in real-time.");
            feedbackCard.classList.add('hidden');
            // Re-analyze to see the update (optional, but shows the effect)
            analyzeSentiment();
        } else {
            alert(data.message || "Feedback failed");
        }
    } catch (err) {
        alert("Server error");
    } finally {
        buttons.forEach(b => b.disabled = false);
    }
}

function updateResult(id, result) {
    const container = document.getElementById(id);
    let icon = 'fa-meh';
    if (result.sentiment === 'Positive') icon = 'fa-smile';
    if (result.sentiment === 'Negative') icon = 'fa-frown';

    const sourceTag = result.source === 'User Feedback'
        ? `<span style="display:block; font-size: 0.75rem; color: var(--success); margin-bottom: 0.25rem;"><i class="fas fa-check-circle"></i> User Feedback Applied</span>`
        : '';

    container.innerHTML = `
        ${sourceTag}
        <span class="sentiment-badge sentiment-${result.sentiment}">
            <i class="fas ${icon}"></i> ${result.sentiment}
        </span>
        <p style="font-size: 0.9rem; color: var(--text-muted); padding-left: 0.5rem;">Confidence Index: ${(result.confidence * 100).toFixed(1)}%</p>
    `;
}

// Chart Visualization
function initChart() {
    const ctx = document.getElementById('accuracyChart').getContext('2d');
    if (accuracyChart) accuracyChart.destroy();

    accuracyChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Naive Bayes', 'SVM Classifier'],
            datasets: [{
                label: 'Model Accuracy',
                data: [0, 0],
                backgroundColor: ['#c084fc', '#818cf8'],
                borderRadius: 12,
                borderSkipped: false,
                barThickness: 40
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1,
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    ticks: { color: '#94a3b8', font: { size: 10 } }
                },
                x: {
                    grid: { display: false },
                    ticks: { color: '#94a3b8', font: { weight: '600', size: 11 } }
                }
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: 'rgba(15, 23, 42, 0.9)',
                    titleFont: { family: 'Outfit', size: 14 },
                    bodyFont: { family: 'Inter', size: 12 },
                    padding: 12,
                    cornerRadius: 8,
                    displayColors: false
                }
            }
        }
    });
}

function updateChart(metrics) {
    if (!accuracyChart) return;
    accuracyChart.data.datasets[0].data = [metrics.nb_accuracy, metrics.svm_accuracy];
    accuracyChart.update();
}

// --- NEW: Dataset Management Logic ---

const fileInput = document.getElementById('file-input');
const dropZone = document.getElementById('drop-zone');

if (fileInput) {
    fileInput.addEventListener('change', (e) => handleFileUpload(e.target.files[0]));
}

if (dropZone) {
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(evt => {
        dropZone.addEventListener(evt, (e) => {
            e.preventDefault();
            e.stopPropagation();
        });
    });

    dropZone.addEventListener('drop', (e) => handleFileUpload(e.dataTransfer.files[0]));
}

async function handleFileUpload(file) {
    if (!file || !file.name.endsWith('.csv')) return alert("Please upload a valid CSV file");

    const formData = new FormData();
    formData.append('file', file);

    const dropZone = document.getElementById('drop-zone');
    const originalHTML = dropZone.innerHTML;
    dropZone.innerHTML = '<i class="fas fa-spinner fa-spin"></i><p>Uploading Dataset...</p>';

    try {
        const response = await fetch(`${API_BASE}/upload-dataset`, {
            method: 'POST',
            body: formData
        });
        const data = await response.json();

        if (response.ok) {
            dropZone.innerHTML = `<i class="fas fa-file-csv" style="color: var(--success)"></i><h3>${file.name}</h3><p>${data.samples} samples detected ready for training.</p>`;
            document.getElementById('train-btn').style.display = 'flex';
            document.getElementById('dataset-stats').style.display = 'grid';
            document.getElementById('stat-samples').textContent = data.samples;
            document.getElementById('training-badge').className = 'sentiment-badge sentiment-Positive';
            document.getElementById('training-badge').innerHTML = '<i class="fas fa-file-import"></i> Loaded';

            // Render Distribution Chart
            if (data.distribution) {
                document.getElementById('distribution-chart-container').style.display = 'block';
                renderDistChart(data.distribution);
            }
        } else {
            alert(data.detail || "Upload failed");
            dropZone.innerHTML = originalHTML;
        }
    } catch (err) {
        alert("Server error during upload");
        dropZone.innerHTML = originalHTML;
    }
}

function renderDistChart(distribution) {
    const ctx = document.getElementById('distChart').getContext('2d');
    if (distChart) distChart.destroy();

    const labels = Object.keys(distribution);
    const counts = Object.values(distribution);

    const colors = {
        'Positive': '#10b981',
        'Negative': '#ef4444',
        'Neutral': '#94a3b8'
    };

    const backgroundColors = labels.map(l => colors[l] || '#818cf8');

    distChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Samples',
                data: counts,
                backgroundColor: backgroundColors,
                borderRadius: 10,
                barThickness: 32
            }]
        },
        options: {
            indexAxis: 'y', // Makes it a horizontal bar chart
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    beginAtZero: true,
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    ticks: { color: '#94a3b8', font: { size: 10 } }
                },
                y: {
                    grid: { display: false },
                    ticks: { color: '#f8fafc', font: { weight: '600', size: 11 } }
                }
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: 'rgba(15, 23, 42, 0.9)',
                    titleFont: { family: 'Outfit', size: 14 },
                    bodyFont: { family: 'Inter', size: 12 },
                    padding: 12,
                    cornerRadius: 8
                }
            }
        }
    });
}

async function startCustomTraining() {
    const btn = document.getElementById('train-btn');
    const progressDiv = document.getElementById('training-progress');

    btn.style.display = 'none';
    progressDiv.style.display = 'block';

    try {
        const response = await fetch(`${API_BASE}/train-custom`, { method: 'POST' });
        if (response.ok) {
            pollTrainingStatus();
        } else {
            alert("Could not start training");
            btn.style.display = 'flex';
        }
    } catch (err) {
        alert("Connection error");
        btn.style.display = 'flex';
    }
}

let pollInterval = null;
function pollTrainingStatus() {
    if (pollInterval) clearInterval(pollInterval);

    pollInterval = setInterval(async () => {
        try {
            const response = await fetch(`${API_BASE}/training-status`);
            const data = await response.json();

            const progressBar = document.getElementById('train-progress-bar');
            const progressText = document.getElementById('progress-text');
            const badge = document.getElementById('training-badge');

            if (data.status === 'training') {
                progressBar.style.width = `${data.progress}%`;
                progressText.textContent = `Optimizing Brain... ${data.progress}%`;
                badge.className = 'sentiment-badge sentiment-Neutral';
                badge.innerHTML = '<i class="fas fa-sync fa-spin"></i> Training';
            } else if (data.status === 'completed') {
                clearInterval(pollInterval);
                progressBar.style.width = '100%';
                progressText.textContent = 'Training Complete! Model synchronized.';
                badge.className = 'sentiment-badge sentiment-Positive';
                badge.innerHTML = '<i class="fas fa-graduation-cap"></i> Smarter';

                document.getElementById('stat-accuracy').textContent = `${(data.accuracy * 100).toFixed(1)}%`;

                // Refresh main accuracy chart if metrics are available elsewhere
                // we'll trigger a dummy analysis or wait for next analyze to update chart fully
                alert("Knowledge Base Updated Successfully! You can now test the new model.");
            } else if (data.status === 'error') {
                clearInterval(pollInterval);
                alert(`Training Error: ${data.error}`);
                document.getElementById('train-btn').style.display = 'flex';
                progressDiv.style.display = 'none';
            }
        } catch (err) {
            console.error("Polling error", err);
        }
    }, 1000);
}

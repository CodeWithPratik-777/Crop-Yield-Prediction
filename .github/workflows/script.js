document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('predictionForm');
    const resultCard = document.getElementById('resultCard');
    const yieldValue = document.getElementById('yieldValue');
    const confidenceFill = document.getElementById('confidenceFill');
    const confidenceText = document.getElementById('confidenceText');
    const recommendationsCard = document.getElementById('recommendationsCard');
    const recommendationsList = document.getElementById('recommendationsList');
    const chartCard = document.getElementById('chartCard');
    const historyList = document.getElementById('historyList');
    const backendUrl = 'http://127.0.0.1:5000/api/predict';

    const savedPredictions = JSON.parse(localStorage.getItem('cropPredictions') || '[]');
    let predictions = Array.isArray(savedPredictions) ? savedPredictions : [];

    function updateHistory() {
        historyList.innerHTML = '';
        if (predictions.length === 0) {
            historyList.innerHTML = '<p class="no-data">No predictions yet</p>';
            return;
        }

        predictions.slice(0, 10).forEach(pred => {
            const time = new Date(pred.timestamp);
            const item = document.createElement('div');
            item.className = 'history-item';
            item.innerHTML = `
                <div class="crop-info">
                    <div class="crop-name">${pred.cropType.toUpperCase()} (${pred.season})</div>
                    <div class="crop-time">${time.toLocaleString()}</div>
                </div>
                <div class="yield-amount">${pred.yield.toFixed(2)} t/ha</div>
            `;
            historyList.appendChild(item);
        });
    }

    function setRangeValue(id, value) {
        const element = document.getElementById(id);
        if (element) element.textContent = value;
    }

    function showResult(prediction) {
        yieldValue.textContent = prediction.yield.toFixed(2);
        confidenceFill.style.width = prediction.confidence + '%';
        confidenceText.textContent = prediction.confidence + '%';
        resultCard.classList.remove('hidden');
    }

    function showRecommendations(data) {
        const recommendations = [];
        const optimalTemps = { wheat: 20, rice: 25, corn: 22, soybean: 20, cotton: 25, potato: 18 };
        const tempOptimal = optimalTemps[data.cropType] || 22;
        if (data.temperature < tempOptimal - 5) {
            recommendations.push({ icon: '🌡️', title: 'Temperature Low', text: 'Temperature is below optimal. Use mulching or cover crops.' });
        } else if (data.temperature > tempOptimal + 5) {
            recommendations.push({ icon: '🌡️', title: 'Temperature High', text: 'Temperature is above optimal. Increase irrigation.' });
        }

        if (data.humidity < 40) {
            recommendations.push({ icon: '💧', title: 'Low Humidity', text: 'Humidity is low. Increase irrigation and moisture retention.' });
        } else if (data.humidity > 80) {
            recommendations.push({ icon: '💧', title: 'High Humidity', text: 'Humidity is high. Improve ventilation and drainage.' });
        }

        if (data.phLevel < 6) {
            recommendations.push({ icon: '⚗️', title: 'Acidic Soil', text: 'Soil is acidic. Add lime to improve pH.' });
        } else if (data.phLevel > 7.5) {
            recommendations.push({ icon: '⚗️', title: 'Alkaline Soil', text: 'Soil is alkaline. Add sulfur or organic matter.' });
        }

        recommendationsList.innerHTML = '';
        if (recommendations.length === 0) {
            recommendationsList.innerHTML = '<p class="no-data">All selected values are within a reasonable range.</p>';
        } else {
            recommendations.forEach(rec => {
                const container = document.createElement('div');
                container.className = 'recommendation-item';
                container.innerHTML = `
                    <div class="recommendation-icon">${rec.icon}</div>
                    <div class="recommendation-text">
                        <strong>${rec.title}</strong>
                        <small>${rec.text}</small>
                    </div>
                `;
                recommendationsList.appendChild(container);
            });
        }
        recommendationsCard.classList.remove('hidden');
    }

    function displayChart(data) {
        const canvas = document.getElementById('performanceChart');
        const ctx = canvas.getContext('2d');
        chartCard.classList.remove('hidden');
        canvas.width = canvas.offsetWidth;
        canvas.height = 240;
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        const metrics = [
            { label: 'Temp', value: Math.min((data.temperature / 50) * 100, 100) },
            { label: 'Rain', value: Math.min((data.rainfall / 1000) * 100, 100) },
            { label: 'Humidity', value: data.humidity },
            { label: 'pH', value: Math.min((data.phLevel / 14) * 100, 100) }
        ];

        const barWidth = (canvas.width - 40) / metrics.length;
        metrics.forEach((metric, idx) => {
            const x = 20 + idx * (barWidth + 10);
            const y = canvas.height - 30;
            const height = (metric.value / 100) * (canvas.height - 70);
            ctx.fillStyle = '#0a6cff';
            ctx.fillRect(x, y - height, barWidth, height);
            ctx.fillStyle = '#1a1a1a';
            ctx.font = '12px Arial';
            ctx.textAlign = 'center';
            ctx.fillText(metric.label, x + barWidth / 2, canvas.height - 8);
            ctx.fillText(Math.round(metric.value) + '%', x + barWidth / 2, y - height - 8);
        });
    }

    function savePrediction(prediction) {
        predictions.unshift(prediction);
        if (predictions.length > 10) predictions = predictions.slice(0, 10);
        localStorage.setItem('cropPredictions', JSON.stringify(predictions));
        updateHistory();
    }

    function getFormData() {
        return {
            cropType: document.getElementById('cropType').value,
            temperature: parseFloat(document.getElementById('temperature').value),
            rainfall: parseFloat(document.getElementById('rainfall').value),
            humidity: parseFloat(document.getElementById('humidity').value),
            phLevel: parseFloat(document.getElementById('phLevel').value),
            season: document.getElementById('season').value,
            timestamp: new Date().toISOString()
        };
    }

    function validateData(data) {
        return (
            data.cropType &&
            !Number.isNaN(data.temperature) &&
            !Number.isNaN(data.rainfall) &&
            !Number.isNaN(data.humidity) &&
            !Number.isNaN(data.phLevel) &&
            data.season
        );
    }

    form.addEventListener('submit', async (event) => {
        event.preventDefault();
        const data = getFormData();
        if (!validateData(data)) {
            alert('Please fill in all required fields with valid values.');
            return;
        }

        try {
            const response = await fetch(backendUrl, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    temperature: data.temperature,
                    rainfall: data.rainfall,
                    humidity: data.humidity,
                    soil_ph: data.phLevel,
                    seed_variety: data.cropType
                })
            });

            if (!response.ok) {
                const json = await response.json();
                throw new Error(json.error || 'Prediction failed.');
            }

            const result = await response.json();
            const prediction = {
                ...data,
                yield: result.prediction,
                confidence: result.confidence || 75
            };
            showResult(prediction);
            showRecommendations(data);
            displayChart(data);
            savePrediction(prediction);
        } catch (err) {
            alert('Unable to connect to backend: ' + err.message + '. Make sure app.py is running.');
        }
    });
    
        // Hide result/recommendations/chart on form reset
        form.addEventListener('reset', () => {
            if (resultCard) resultCard.classList.add('hidden');
            if (recommendationsCard) recommendationsCard.classList.add('hidden');
            if (chartCard) chartCard.classList.add('hidden');
        });

    const rangeFields = ['temperature', 'rainfall', 'soilMoisture', 'phLevel', 'nitrogen', 'phosphorus', 'potassium', 'pesticide', 'sunlight'];
    rangeFields.forEach(id => {
        const el = document.getElementById(id);
        if (el) {
            el.addEventListener('input', (event) => {
                setRangeValue(id + 'Value', event.target.value);
            });
        }
    });
    
        // Initialize visible values for fields on load
        watchedFields.forEach(field => {
            const input = document.getElementById(field.id);
            if (input) setRangeValue(field.label, input.value || '0');
        });

    updateHistory();
});
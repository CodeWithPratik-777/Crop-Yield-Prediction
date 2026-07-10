// Smart Crop Yield Prediction System - JavaScript

class CropYieldPredictor {
    constructor() {
        this.predictions = [];
        this.form = document.getElementById('predictionForm');
        this.setupEventListeners();
        this.loadPredictions();
    }

    setupEventListeners() {
        // Form submission
        this.form.addEventListener('submit', (e) => {
            e.preventDefault();
            this.predictYield();
        });

        // Update range displays
        document.getElementById('temperature').addEventListener('input', (e) => {
            document.getElementById('tempValue').textContent = e.target.value;
        });
        document.getElementById('rainfall').addEventListener('input', (e) => {
            document.getElementById('rainfallValue').textContent = e.target.value;
        });
        document.getElementById('soilMoisture').addEventListener('input', (e) => {
            document.getElementById('moistureValue').textContent = e.target.value;
        });
        document.getElementById('phLevel').addEventListener('input', (e) => {
            document.getElementById('phValue').textContent = e.target.value;
        });
        document.getElementById('nitrogen').addEventListener('input', (e) => {
            document.getElementById('nitrogenValue').textContent = e.target.value;
        });
        document.getElementById('phosphorus').addEventListener('input', (e) => {
            document.getElementById('phosphorusValue').textContent = e.target.value;
        });
        document.getElementById('potassium').addEventListener('input', (e) => {
            document.getElementById('potassiumValue').textContent = e.target.value;
        });
        document.getElementById('pesticide').addEventListener('input', (e) => {
            document.getElementById('pesticideValue').textContent = e.target.value;
        });
        document.getElementById('sunlight').addEventListener('input', (e) => {
            document.getElementById('sunlightValue').textContent = e.target.value;
        });
    }

    getFormData() {
        return {
            cropType: document.getElementById('cropType').value,
            temperature: parseFloat(document.getElementById('temperature').value),
            rainfall: parseFloat(document.getElementById('rainfall').value),
            soilMoisture: parseFloat(document.getElementById('soilMoisture').value),
            phLevel: parseFloat(document.getElementById('phLevel').value),
            nitrogen: parseFloat(document.getElementById('nitrogen').value),
            phosphorus: parseFloat(document.getElementById('phosphorus').value),
            potassium: parseFloat(document.getElementById('potassium').value),
            pesticide: parseFloat(document.getElementById('pesticide').value),
            sunlight: parseFloat(document.getElementById('sunlight').value),
            season: document.getElementById('season').value,
            timestamp: new Date()
        };
    }

    predictYield() {
        const data = this.getFormData();
        
        // Validate form data
        if (!this.validateFormData(data)) {
            alert('Please fill in all fields correctly.');
            return;
        }

        // Calculate yield based on inputs
        const prediction = this.calculateYield(data);
        
        // Store prediction
        this.predictions.unshift(prediction);
        this.savePredictions();
        
        // Display results
        this.displayResults(prediction);
        this.displayRecommendations(prediction, data);
        this.displayChart(data);
        this.updateHistory();
    }

    validateFormData(data) {
        return (
            data.cropType &&
            data.temperature >= 0 && data.temperature <= 50 &&
            data.rainfall >= 0 && data.rainfall <= 1000 &&
            data.soilMoisture >= 0 && data.soilMoisture <= 100 &&
            data.phLevel >= 0 && data.phLevel <= 14 &&
            data.nitrogen >= 0 && data.nitrogen <= 500 &&
            data.phosphorus >= 0 && data.phosphorus <= 500 &&
            data.potassium >= 0 && data.potassium <= 500 &&
            data.pesticide >= 0 && data.pesticide <= 100 &&
            data.sunlight >= 0 && data.sunlight <= 24 &&
            data.season
        );
    }

    calculateYield(data) {
        // Base yields for different crops (tons/hectare)
        const baseYields = {
            wheat: 3.5,
            rice: 4.5,
            corn: 5.0,
            soybean: 2.5,
            cotton: 1.5,
            potato: 15.0
        };

        let baseYield = baseYields[data.cropType] || 3.5;
        let yield = baseYield;

        // Temperature factor
        const optimalTemp = {
            wheat: 20, rice: 25, corn: 22, 
            soybean: 20, cotton: 25, potato: 18
        };
        const tempOptimal = optimalTemp[data.cropType];
        const tempFactor = 1 - Math.abs(data.temperature - tempOptimal) / 100;
        yield *= Math.max(0.5, tempFactor);

        // Rainfall factor
        const optimalRainfall = {
            wheat: 400, rice: 800, corn: 600,
            soybean: 450, cotton: 600, potato: 500
        };
        const rainfallOptimal = optimalRainfall[data.cropType];
        const rainfallFactor = 1 - Math.abs(data.rainfall - rainfallOptimal) / 1000;
        yield *= Math.max(0.4, rainfallFactor);

        // Soil Moisture factor (optimal: 40-60%)
        const moistureFactor = 1 - Math.abs(data.soilMoisture - 50) / 100;
        yield *= Math.max(0.5, moistureFactor);

        // pH factor (optimal: 6-7 for most crops)
        const phFactor = 1 - Math.abs(data.phLevel - 6.5) / 10;
        yield *= Math.max(0.6, phFactor);

        // Nutrient factor (NPK)
        const totalNutrients = data.nitrogen + data.phosphorus + data.potassium;
        const optimalNutrients = {
            wheat: 260, rice: 280, corn: 300,
            soybean: 200, cotton: 250, potato: 350
        };
        const nutrientOptimal = optimalNutrients[data.cropType];
        const nutrientFactor = totalNutrients > 0 ? 
            1 - Math.abs(totalNutrients - nutrientOptimal) / 500 : 0.5;
        yield *= Math.max(0.4, nutrientFactor);

        // Pesticide factor (slight improvement, then diminishing returns)
        const pesticideFactor = 1 + Math.min(data.pesticide / 50, 0.3) * (1 - data.pesticide / 200);
        yield *= Math.max(0.8, pesticideFactor);

        // Sunlight factor (optimal: 8-12 hours)
        const sunlightFactor = 1 - Math.abs(data.sunlight - 10) / 20;
        yield *= Math.max(0.4, sunlightFactor);

        // Season adjustment
        const seasonFactors = {
            kharif: 0.95,
            rabi: 1.0,
            summer: 0.85
        };
        yield *= seasonFactors[data.season] || 1.0;

        // Calculate confidence based on how close values are to optimal
        let confidence = 80;
        confidence -= Math.abs(data.temperature - tempOptimal) / 100 * 15;
        confidence -= Math.abs(data.rainfall - rainfallOptimal) / 1000 * 15;
        confidence = Math.max(40, Math.min(98, confidence));

        return {
            crop: data.cropType,
            yield: Math.round(yield * 100) / 100,
            confidence: Math.round(confidence),
            data: data
        };
    }

    displayResults(prediction) {
        const resultCard = document.getElementById('resultCard');
        const yieldValue = document.getElementById('yieldValue');
        const confidenceFill = document.getElementById('confidenceFill');
        const confidenceText = document.getElementById('confidenceText');

        yieldValue.textContent = prediction.yield.toFixed(2);
        confidenceFill.style.width = prediction.confidence + '%';
        confidenceText.textContent = prediction.confidence + '%';

        resultCard.classList.remove('hidden');
    }

    displayRecommendations(prediction, data) {
        const recommendationsCard = document.getElementById('recommendationsCard');
        const recommendationsList = document.getElementById('recommendationsList');
        const recommendations = [];

        // Temperature recommendations
        const optimalTemps = {
            wheat: 20, rice: 25, corn: 22,
            soybean: 20, cotton: 25, potato: 18
        };
        if (data.temperature < optimalTemps[data.cropType] - 5) {
            recommendations.push({
                icon: '🌡️',
                title: 'Temperature Low',
                text: 'Current temperature is below optimal. Consider protective measures like mulching.'
            });
        } else if (data.temperature > optimalTemps[data.cropType] + 5) {
            recommendations.push({
                icon: '🌡️',
                title: 'Temperature High',
                text: 'Temperature is higher than optimal. Increase irrigation frequency.'
            });
        }

        // Rainfall recommendations
        const optimalRainfalls = {
            wheat: 400, rice: 800, corn: 600,
            soybean: 450, cotton: 600, potato: 500
        };
        if (data.rainfall < optimalRainfalls[data.cropType] - 100) {
            recommendations.push({
                icon: '💧',
                title: 'Insufficient Rainfall',
                text: 'Rainfall is below optimal. Increase irrigation and water management.'
            });
        } else if (data.rainfall > optimalRainfalls[data.cropType] + 100) {
            recommendations.push({
                icon: '💧',
                title: 'Excess Rainfall',
                text: 'Rainfall is too high. Ensure proper drainage systems.'
            });
        }

        // Soil Moisture recommendations
        if (data.soilMoisture < 30) {
            recommendations.push({
                icon: '🌵',
                title: 'Low Soil Moisture',
                text: 'Soil is too dry. Increase watering schedule immediately.'
            });
        } else if (data.soilMoisture > 70) {
            recommendations.push({
                icon: '💦',
                title: 'High Soil Moisture',
                text: 'Soil is too wet. Allow drainage and reduce watering.'
            });
        }

        // pH recommendations
        if (data.phLevel < 5.5) {
            recommendations.push({
                icon: '⚗️',
                title: 'Low pH',
                text: 'Soil is too acidic. Apply lime to increase pH levels.'
            });
        } else if (data.phLevel > 7.5) {
            recommendations.push({
                icon: '⚗️',
                title: 'High pH',
                text: 'Soil is too alkaline. Add sulfur-based amendments.'
            });
        }

        // Nutrient recommendations
        if (data.nitrogen < 50) {
            recommendations.push({
                icon: '🧪',
                title: 'Low Nitrogen',
                text: 'Increase nitrogen fertilizer application for better growth.'
            });
        }
        if (data.phosphorus < 30) {
            recommendations.push({
                icon: '🧪',
                title: 'Low Phosphorus',
                text: 'Phosphorus levels are low. Apply phosphorus fertilizer.'
            });
        }
        if (data.potassium < 40) {
            recommendations.push({
                icon: '🧪',
                title: 'Low Potassium',
                text: 'Potassium levels are low. Apply potassium fertilizer.'
            });
        }

        // Sunlight recommendations
        if (data.sunlight < 6) {
            recommendations.push({
                icon: '☀️',
                title: 'Insufficient Sunlight',
                text: 'Field receives less sunlight. Consider removing shade-causing elements.'
            });
        }

        // Pesticide recommendations
        if (data.pesticide < 10) {
            recommendations.push({
                icon: '🛡️',
                title: 'Low Pest Control',
                text: 'Consider increasing pesticide application to prevent crop damage.'
            });
        } else if (data.pesticide > 50) {
            recommendations.push({
                icon: '🛡️',
                title: 'High Pesticide Usage',
                text: 'Pesticide levels are high. Practice integrated pest management.'
            });
        }

        // General recommendation based on yield
        if (prediction.yield < 2) {
            recommendations.push({
                icon: '⚠️',
                title: 'Low Expected Yield',
                text: 'Expected yield is below average. Review all input parameters.'
            });
        } else if (prediction.yield > 5) {
            recommendations.push({
                icon: '✅',
                title: 'Excellent Conditions',
                text: 'All conditions appear optimal. Maintain current practices.'
            });
        }

        // Display recommendations
        recommendationsList.innerHTML = '';
        if (recommendations.length === 0) {
            recommendationsList.innerHTML = '<p class="no-data">All conditions are optimal. No specific recommendations.</p>';
        } else {
            recommendations.forEach(rec => {
                const recElement = document.createElement('div');
                recElement.className = 'recommendation-item';
                recElement.innerHTML = `
                    <div class="recommendation-icon">${rec.icon}</div>
                    <div class="recommendation-text">
                        <strong>${rec.title}</strong>
                        <small>${rec.text}</small>
                    </div>
                `;
                recommendationsList.appendChild(recElement);
            });
        }

        document.getElementById('recommendationsCard').classList.remove('hidden');
    }

    displayChart(data) {
        const canvas = document.getElementById('performanceChart');
        const chartCard = document.getElementById('chartCard');
        
        // Simple chart using canvas
        const ctx = canvas.getContext('2d');
        canvas.width = canvas.offsetWidth;
        canvas.height = 250;

        // Clear canvas
        ctx.fillStyle = '#f5f7fa';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        // Draw chart
        const metrics = [
            { label: 'Temp', value: Math.min(data.temperature / 50 * 100, 100) },
            { label: 'Rain', value: Math.min(data.rainfall / 1000 * 100, 100) },
            { label: 'Moisture', value: data.soilMoisture },
            { label: 'pH', value: Math.min((data.phLevel / 14) * 100, 100) },
            { label: 'NPK', value: Math.min((data.nitrogen + data.phosphorus + data.potassium) / 15, 100) },
            { label: 'Sunlight', value: Math.min((data.sunlight / 24) * 100, 100) }
        ];

        const barWidth = canvas.width / (metrics.length * 2);
        const maxHeight = canvas.height - 40;

        metrics.forEach((metric, index) => {
            const x = index * barWidth * 2 + barWidth / 2;
            const height = (metric.value / 100) * maxHeight;
            const y = canvas.height - height - 20;

            // Draw bar
            ctx.fillStyle = `hsl(${120 - metric.value * 1.2}, 70%, 50%)`;
            ctx.fillRect(x, y, barWidth, height);

            // Draw label
            ctx.fillStyle = '#2c3e50';
            ctx.font = '12px Arial';
            ctx.textAlign = 'center';
            ctx.fillText(metric.label, x + barWidth / 2, canvas.height - 5);

            // Draw percentage
            ctx.fillStyle = '#27ae60';
            ctx.font = 'bold 12px Arial';
            ctx.fillText(Math.round(metric.value) + '%', x + barWidth / 2, y - 5);
        });

        chartCard.classList.remove('hidden');
    }

    updateHistory() {
        const historyList = document.getElementById('historyList');
        historyList.innerHTML = '';

        if (this.predictions.length === 0) {
            historyList.innerHTML = '<p class="no-data">No predictions yet</p>';
            return;
        }

        this.predictions.slice(0, 10).forEach(pred => {
            const time = new Date(pred.data.timestamp);
            const timeString = time.toLocaleString();

            const historyItem = document.createElement('div');
            historyItem.className = 'history-item';
            historyItem.innerHTML = `
                <div class="crop-info">
                    <div class="crop-name">${pred.crop.toUpperCase()} (${pred.data.season})</div>
                    <div class="crop-time">${timeString}</div>
                </div>
                <div class="yield-amount">${pred.yield.toFixed(2)} t/ha</div>
            `;
            historyList.appendChild(historyItem);
        });
    }

    savePredictions() {
        localStorage.setItem('cropPredictions', JSON.stringify(this.predictions));
    }

    loadPredictions() {
        const saved = localStorage.getItem('cropPredictions');
        if (saved) {
            try {
                this.predictions = JSON.parse(saved);
                this.updateHistory();
            } catch (e) {
                console.error('Error loading predictions:', e);
                this.predictions = [];
            }
        }
    }
}

// Initialize the predictor when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    new CropYieldPredictor();
});
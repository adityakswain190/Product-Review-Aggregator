

document.addEventListener('DOMContentLoaded', function() {
    // Form submission with loading indicator
    const analyzeForm = document.getElementById('analyze-form');
    const loadingIndicator = document.getElementById('loading-indicator');
    
    if (analyzeForm) {
        analyzeForm.addEventListener('submit', function() {
            loadingIndicator.style.display = 'block';
        });
    }
    
    // Initialize tooltips
    const tooltipTriggerList = document.querySelectorAll('[data-bs-toggle="tooltip"]')
    const tooltipList = [...tooltipTriggerList].map(tooltipTriggerEl => new bootstrap.Tooltip(tooltipTriggerEl))
    
    // URL validator function
    const productUrlInput = document.getElementById('product_url');
    if (productUrlInput) {
        productUrlInput.addEventListener('input', function() {
            const url = this.value.trim();
            const isAmazonUrl = /amazon\.(com|in|co\.uk|de|fr|es|it|co\.jp|ca|com\.au)/.test(url);
            
            if (url && !isAmazonUrl) {
                this.classList.add('is-invalid');
                document.getElementById('url-feedback').style.display = 'block';
            } else {
                this.classList.remove('is-invalid');
                document.getElementById('url-feedback').style.display = 'none';
            }
        });
    }
    
    // Feature highlights animation
    const featureCards = document.querySelectorAll('.feature-card');
    
    if (featureCards.length > 0) {
        featureCards.forEach((card, index) => {
            setTimeout(() => {
                card.classList.add('animate__animated', 'animate__fadeInUp');
            }, index * 200);
        });
    }
    
    // Results page - animate sentiment meter
    const sentimentMeter = document.getElementById('sentiment-meter');
    const sentimentMarker = document.getElementById('sentiment-marker');
    
    if (sentimentMeter && sentimentMarker) {
        const sentimentValue = parseFloat(sentimentMeter.dataset.sentiment);
        // Convert sentiment from [-1, 1] to [0, 100] for positioning
        const position = ((sentimentValue + 1) / 2) * 100;
        
        // Animate the marker
        setTimeout(() => {
            sentimentMarker.style.left = `${position}%`;
        }, 500);
    }
});

/**
 * Generate Page JavaScript - Virtual Fashion Try-On
 * Optimized for performance with minimal DOM operations
 */

document.addEventListener('DOMContentLoaded', function() {
    console.log('Generate Page Loaded');
    initGeneratePage();
});

/**
 * Initialize the generate page functionality
 */
function initGeneratePage() {
    setupFileUpload();
    checkPendingResults();
    setupFormSubmission();
}

/**
 * Set up file upload interactions
 */
function setupFileUpload() {
    const fileUpload = document.getElementById('imageUpload');
    const fileWrapper = document.querySelector('.file-upload-wrapper');
    const filePlaceholder = document.querySelector('.file-upload-placeholder');

    if (!fileUpload || !fileWrapper || !filePlaceholder) return;

    // File selection event
    fileUpload.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file) {
            updateFileUploadUI(file, filePlaceholder, fileWrapper);
        }
    });

    // Drag and drop functionality
    fileWrapper.addEventListener('dragover', function(e) {
        e.preventDefault();
        this.style.borderColor = 'var(--accent-primary)';
        this.style.background = 'var(--glass-primary)';
        this.style.transform = 'scale(1.02)';
    });

    fileWrapper.addEventListener('dragleave', function(e) {
        e.preventDefault();
        this.style.borderColor = 'var(--glass-border-secondary)';
        this.style.background = 'var(--glass-secondary)';
        this.style.transform = 'scale(1)';
    });

    fileWrapper.addEventListener('drop', function(e) {
        e.preventDefault();
        this.style.borderColor = 'var(--glass-border-secondary)';
        this.style.background = 'var(--glass-secondary)';
        this.style.transform = 'scale(1)';
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            fileUpload.files = files;
            fileUpload.dispatchEvent(new Event('change'));
        }
    });
}

/**
 * Update the file upload UI after file selection
 */
function updateFileUploadUI(file, filePlaceholder, fileWrapper) {
    filePlaceholder.innerHTML = `
        <i class="bi bi-check-circle-fill" style="font-size: 3rem; color: #4ade80; margin-bottom: 1rem; display: block;"></i>
        <span style="color: var(--text-primary); font-size: 1.2rem; font-weight: 500; display: block; margin-bottom: 0.5rem;">${file.name}</span>
        <p style="color: var(--text-muted); margin: 0; font-size: 0.9rem;">Ready to upload</p>
    `;
    fileWrapper.style.borderColor = '#4ade80';
    fileWrapper.style.background = 'rgba(74, 222, 128, 0.1)';
}

/**
 * Set up the form submission handling
 */
function setupFormSubmission() {
    const uploadForm = document.getElementById('uploadForm');
    if (!uploadForm) return;

    uploadForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const submitBtn = document.getElementById('submitBtn');
        const progressContainer = document.getElementById('progressContainer');
        const uploadContainer = document.querySelector('.upload-form-container');
        const formData = new FormData(this);
        
        // Show loading state
        if (submitBtn) {
            submitBtn.disabled = true;
            submitBtn.innerHTML = '<i class="bi bi-hourglass-split me-2"></i>Processing...';
        }
        
        // Hide form and show progress
        if (uploadContainer) uploadContainer.style.display = 'none';
        if (progressContainer) progressContainer.style.display = 'block';
        
        // Submit the form
        uploadAsync(formData);
    });
}

/**
 * Asynchronously upload the form data
 */
function uploadAsync(formData) {
    const csrfToken = document.querySelector('[name=csrfmiddlewaretoken]')?.value;
    if (!csrfToken) {
        showError('CSRF token not found. Please refresh the page and try again.');
        return;
    }

    fetch('/upload_async/', {
        method: 'POST',
        body: formData,
        headers: {
            'X-CSRFToken': csrfToken
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // Store result ID and start monitoring
            localStorage.setItem('pendingResultId', data.result_id);
            monitorProgress(data.result_id);
        } else {
            showError(data.error || 'Unknown error occurred');
        }
    })
    .catch(error => {
        console.error('Upload failed:', error);
        showError('Upload failed. Please try again.');
    });
}

/**
 * Monitor the progress of the generation
 */
function monitorProgress(resultId) {
    const progressMessage = document.getElementById('progressMessage');
    if (progressMessage) {
        progressMessage.textContent = 'Your AI-powered try-on is being generated...';
    }
    
    let pollCount = 0;
    const maxPolls = 120; // 4 minutes max (120 * 2 seconds)
    
    const checkInterval = setInterval(() => {
        pollCount++;
        
        // Safety timeout
        if (pollCount > maxPolls) {
            clearInterval(checkInterval);
            if (progressMessage) {
                progressMessage.innerHTML = `
                    Processing is taking longer than expected. 
                    <br><br>
                    <button onclick="window.location.href='/result/${resultId}/'" 
                            style="background: var(--gradient-primary); border: none; padding: 10px 20px; border-radius: 8px; color: white; cursor: pointer; margin-top: 10px;">
                        Check Result Page
                    </button>
                `;
            }
            return;
        }

        // Check if processing is complete
        checkResultStatus(resultId, checkInterval);
    }, 2000); // Check every 2 seconds
}

/**
 * Check the status of a result
 */
function checkResultStatus(resultId, checkInterval) {
    fetch(`/result/${resultId}/`, {
        method: 'GET',
        headers: {'X-Requested-With': 'XMLHttpRequest'}
    })
    .then(response => response.text())
    .then(html => {
        const parser = new DOMParser();
        const doc = parser.parseFromString(html, 'text/html');
        
        const statusBadge = doc.querySelector('.badge');
        const resultImage = doc.querySelector('img[alt*="Try-on result"]');
        const downloadButton = doc.querySelector('a[download]');
        
        let isCompleted = false;
        let isFailed = false;
        
        // Check status badge
        if (statusBadge) {
            const statusText = statusBadge.textContent.trim();
            const hasSuccessClass = statusBadge.classList.contains('bg-success');
            
            if (statusText.includes('Success') || statusText.includes('Completed') || hasSuccessClass) {
                isCompleted = true;
            } else if (statusText.includes('Failed') || statusBadge.classList.contains('bg-danger')) {
                isFailed = true;
            }
        }
        
        // Additional check for result image and download button
        if (resultImage && downloadButton) {
            isCompleted = true;
        }
        
        if (isCompleted) {
            handleCompletedResult(resultId, checkInterval);
        } else if (isFailed) {
            handleFailedResult(checkInterval);
        }
    })
    .catch(error => {
        console.error('Error checking result:', error);
    });
}

/**
 * Handle a completed result
 */
function handleCompletedResult(resultId, checkInterval) {
    clearInterval(checkInterval);
    localStorage.removeItem('pendingResultId');
    
    const progressMessage = document.getElementById('progressMessage');
    if (progressMessage) {
        progressMessage.textContent = 'Generation complete! Redirecting...';
        progressMessage.style.color = '#4ade80';
    }
    
    // Redirect to result page
    setTimeout(() => {
        window.location.href = `/result/${resultId}/`;
    }, 300);
}

/**
 * Handle a failed result
 */
function handleFailedResult(checkInterval) {
    clearInterval(checkInterval);
    localStorage.removeItem('pendingResultId');
    showError('Processing failed. Please try again.');
}

/**
 * Show an error message
 */
function showError(message) {
    const progressMessage = document.getElementById('progressMessage');
    if (progressMessage) {
        progressMessage.textContent = `Error: ${message}`;
        progressMessage.style.color = '#ff6b6b';
    }
    
    // Show retry option after error
    setTimeout(() => {
        const progressContainer = document.getElementById('progressContainer');
        const uploadContainer = document.querySelector('.upload-form-container');
        
        if (progressContainer) progressContainer.style.display = 'none';
        if (uploadContainer) uploadContainer.style.display = 'block';
        
        // Reset form
        const submitBtn = document.getElementById('submitBtn');
        if (submitBtn) {
            submitBtn.disabled = false;
            submitBtn.innerHTML = '<i class="bi bi-magic me-2"></i>Generate Virtual Try-On';
        }
    }, 3000);
}

/**
 * Check for pending results and handle them
 */
function checkPendingResults() {
    const pendingResultId = localStorage.getItem('pendingResultId');
    if (!pendingResultId) return;

    console.log('Checking pending result:', pendingResultId);

    // Check if result is completed or still processing
    fetch(`/result/${pendingResultId}/`, {
        method: 'GET',
        headers: {'X-Requested-With': 'XMLHttpRequest'}
    })
    .then(response => response.text())
    .then(html => {
        const parser = new DOMParser();
        const doc = parser.parseFromString(html, 'text/html');
        
        const statusBadge = doc.querySelector('.status-badge');
        const resultImage = doc.querySelector('img[alt*="Try-on result"]');
        const downloadButton = doc.querySelector('a[download]');
        
        let isCompleted = false;
        let isFailed = false;
        let isProcessing = false;
        
        // Check status badge with the correct class names
        if (statusBadge) {
            const statusText = statusBadge.textContent.trim();
            
            if (statusText.includes('Completed') || statusBadge.classList.contains('status-completed')) {
                isCompleted = true;
            } else if (statusText.includes('Failed') || statusBadge.classList.contains('status-failed')) {
                isFailed = true;
            } else if (statusText.includes('Processing') || statusBadge.classList.contains('status-processing')) {
                isProcessing = true;
            }
        }
        
        // Additional check for result image and download button
        if (resultImage && downloadButton) {
            isCompleted = true;
        }
        
        if (isCompleted) {
            // Auto-redirect to completed result
            console.log('Result completed, redirecting...');
            localStorage.removeItem('pendingResultId');
            window.location.href = `/result/${pendingResultId}/`;
        } else if (isProcessing) {
            // Show progress for processing result
            console.log('Result still processing, showing progress...');
            const uploadContainer = document.querySelector('.upload-form-container');
            const progressContainer = document.getElementById('progressContainer');
            const progressMessage = document.getElementById('progressMessage');
            
            if (uploadContainer) uploadContainer.style.display = 'none';
            if (progressContainer) progressContainer.style.display = 'block';
            if (progressMessage) {
                progressMessage.innerHTML = 'Resuming progress tracking... <br><small>Your generation is still being processed.</small>';
            }
            
            // Continue monitoring from where we left off
            monitorProgress(pendingResultId);
        } else if (isFailed) {
            // Clear failed result and allow new generation
            console.log('Result failed, clearing storage...');
            localStorage.removeItem('pendingResultId');
            showError('Previous generation failed. You can start a new one.');
        }
    })
    .catch(error => {
        console.error('Error checking pending result:', error);
        // Don't clear storage on network errors, might be temporary
        console.log('Network error, will retry...');
    });
}

/**
 * Use a template for the clothing description
 */
function useTemplate(templateText) {
    const clothingDescription = document.getElementById('clothingDescription');
    if (!clothingDescription) return;
    
    clothingDescription.value = templateText;
    clothingDescription.focus();
    
    // Add visual feedback
    clothingDescription.style.borderColor = '#4ade80';
    clothingDescription.style.backgroundColor = 'rgba(74, 222, 128, 0.1)';
    
    setTimeout(() => {
        clothingDescription.style.borderColor = '';
        clothingDescription.style.backgroundColor = '';
    }, 1500);
    
    // Scroll to form
    clothingDescription.scrollIntoView({ 
        behavior: 'smooth', 
        block: 'center' 
    });
}

// Make the useTemplate function globally available
window.useTemplate = useTemplate;

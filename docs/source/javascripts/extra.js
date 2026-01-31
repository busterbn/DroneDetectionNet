/* =============================================================================
   Drone Detector MLOps - Custom JavaScript
   ============================================================================= */

// Wait for DOM to be ready
document.addEventListener("DOMContentLoaded", function() {
  // Add any custom JavaScript functionality here

  // Example: Add external link indicators
  addExternalLinkIndicators();

  // Example: Initialize any custom components
  initCustomComponents();
});

/* -----------------------------------------------------------------------------
   External Link Indicators
   ----------------------------------------------------------------------------- */

function addExternalLinkIndicators() {
  // Get all links in the content area
  const contentLinks = document.querySelectorAll('.md-content a[href^="http"]');

  contentLinks.forEach(function(link) {
    // Check if link is external (not to our own domain)
    if (link.hostname !== window.location.hostname) {
      // Add external link attributes for security
      link.setAttribute('target', '_blank');
      link.setAttribute('rel', 'noopener noreferrer');

      // Optionally add a visual indicator (uncomment if desired)
      // link.classList.add('external-link');
    }
  });
}

/* -----------------------------------------------------------------------------
   Custom Component Initialization
   ----------------------------------------------------------------------------- */

function initCustomComponents() {
  // Add smooth scrolling for anchor links
  document.querySelectorAll('a[href^="#"]').forEach(function(anchor) {
    anchor.addEventListener('click', function(e) {
      const href = this.getAttribute('href');
      if (href !== '#') {
        const target = document.querySelector(href);
        if (target) {
          e.preventDefault();
          target.scrollIntoView({
            behavior: 'smooth',
            block: 'start'
          });
          // Update URL without jumping
          history.pushState(null, null, href);
        }
      }
    });
  });
}

/* -----------------------------------------------------------------------------
   Copy Code Button Enhancement (optional)
   ----------------------------------------------------------------------------- */

// The Material theme already includes copy functionality,
// but you can enhance it here if needed

/* -----------------------------------------------------------------------------
   Analytics Event Tracking (optional)
   ----------------------------------------------------------------------------- */

// Uncomment and customize if you want to track custom events
/*
function trackEvent(category, action, label) {
  if (typeof gtag !== 'undefined') {
    gtag('event', action, {
      'event_category': category,
      'event_label': label
    });
  }
}

// Example: Track code copy events
document.addEventListener('click', function(e) {
  if (e.target.closest('.md-clipboard')) {
    trackEvent('Code', 'copy', window.location.pathname);
  }
});
*/

/* -----------------------------------------------------------------------------
   Search Enhancement (optional)
   ----------------------------------------------------------------------------- */

// You can add search analytics or custom search behavior here

/* -----------------------------------------------------------------------------
   Keyboard Shortcuts (optional)
   ----------------------------------------------------------------------------- */

// Uncomment to add custom keyboard shortcuts
/*
document.addEventListener('keydown', function(e) {
  // Press '/' to focus search (already built into Material)
  // Add your custom shortcuts here
});
*/

// Privacy-respecting analytics via Plausible
// GDPR compliant, no cookies, lightweight
//
// To enable:
// 1. Create a Plausible account at https://plausible.io
// 2. Add your domain to Plausible
// 3. Replace 'mantis.example.com' below with your actual domain
// 4. Uncomment the script injection code
//
// This file is loaded by mkdocs.yml extra_javascript configuration

(function() {
  'use strict';

  // Configuration - set to true and configure domain to enable
  var ANALYTICS_ENABLED = false;
  var PLAUSIBLE_DOMAIN = 'mantis.example.com';  // Replace with your domain

  if (!ANALYTICS_ENABLED) {
    return;
  }

  // Inject Plausible script
  var script = document.createElement('script');
  script.defer = true;
  script.setAttribute('data-domain', PLAUSIBLE_DOMAIN);
  script.src = 'https://plausible.io/js/script.js';
  document.head.appendChild(script);
})();

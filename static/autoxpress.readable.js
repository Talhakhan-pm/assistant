    // Vehicle makes and models data - updated inventory
    const vehicleData = {
      "Acura": ["ILX", "MDX", "RDX", "TLX", "NSX", "RLX", "TSX"],
      "Audi": ["A3", "A4", "A5", "A6", "A7", "A8", "Q3", "Q5", "Q7", "Q8", "e-tron", "TT", "R8"],
      "BMW": ["3-Series", "5-Series", "7-Series", "X1", "X3", "X5", "X7", "M3", "M5", "Z4", "i4", "i7"],
      "Buick": ["Enclave", "Encore", "Envision", "LaCrosse", "Regal"],
      "Cadillac": ["CT4", "CT5", "CTS", "Escalade", "XT4", "XT5", "XT6", "XTS"],
      "Chevrolet": ["Silverado", "Equinox", "Tahoe", "Malibu", "Traverse", "Colorado", "Camaro", "Suburban", "Impala", "Blazer", "Corvette", "Trax", "Spark", "Bolt", "Cruze"],
      "Chevy": ["Silverado", "Equinox", "Tahoe", "Malibu", "Traverse", "Colorado", "Camaro", "Suburban", "Impala", "Blazer", "Corvette", "Trax", "Spark", "Bolt", "Cruze"],
      "Chrysler": ["300", "Pacifica", "Town & Country", "200"],
      "Dodge": ["Challenger", "Charger", "Durango", "Journey", "Grand Caravan", "Ram", "Dart"],
      "Ford": ["F-150", "Mustang", "Explorer", "Escape", "Edge", "Ranger", "Bronco", "Expedition", "Focus", "Fusion", "Taurus", "EcoSport", "Maverick", "F-250", "F-350", "Fiesta"],
      "GMC": ["Sierra", "Terrain", "Yukon", "Acadia", "Canyon", "Savana"],
      "Honda": ["Accord", "Civic", "CR-V", "Pilot", "Odyssey", "HR-V", "Ridgeline", "Fit", "Clarity", "Insight", "Passport"],
      "Hyundai": ["Elantra", "Sonata", "Tucson", "Santa Fe", "Kona", "Palisade", "Veloster", "Venue", "Ioniq", "Santa Cruz", "Accent"],
      "Infiniti": ["Q50", "Q60", "QX50", "QX60", "QX80", "G37", "FX35"],
      "Jaguar": ["F-Pace", "E-Pace", "XE", "XF", "F-Type", "I-Pace", "XJ"],
      "Jeep": ["Grand Cherokee", "Cherokee", "Wrangler", "Compass", "Renegade", "Gladiator", "Patriot", "Liberty", "Wagoneer"],
      "Kia": ["Forte", "Optima", "Sorento", "Sportage", "Telluride", "Soul", "K5", "Stinger", "Carnival", "Rio", "Seltos", "Niro"],
      "Lexus": ["ES", "IS", "RX", "NX", "GX", "LX", "RC", "UX", "LS", "LC"],
      "Lincoln": ["Navigator", "Aviator", "Corsair", "Nautilus", "MKZ", "Continental", "MKC", "MKX"],
      "Mazda": ["Mazda3", "Mazda6", "CX-5", "CX-9", "CX-30", "MX-5 Miata", "CX-3", "CX-50"],
      "Mercedes": ["C-Class", "E-Class", "S-Class", "GLC", "GLE", "GLS", "A-Class", "CLA", "GLB", "G-Class"],
      "Nissan": ["Altima", "Rogue", "Sentra", "Pathfinder", "Frontier", "Murano", "Versa", "Maxima", "Kicks", "Armada", "Titan", "Juke"],
      "Porsche": ["911", "Cayenne", "Macan", "Panamera", "Taycan", "Boxster", "Cayman"],
      "Ram": ["1500", "2500", "3500", "ProMaster", "ProMaster City"],
      "Subaru": ["Outback", "Forester", "Crosstrek", "Impreza", "Legacy", "Ascent", "WRX", "BRZ"],
      "Tesla": ["Model 3", "Model S", "Model X", "Model Y", "Cybertruck"],
      "Toyota": ["Camry", "Corolla", "RAV4", "Tacoma", "Highlander", "4Runner", "Prius", "Tundra", "Sienna", "Avalon", "Land Cruiser", "Sequoia", "Venza", "C-HR", "86", "GR86", "Supra"],
      "Volkswagen": ["Jetta", "Passat", "Tiguan", "Atlas", "Golf", "ID.4", "Taos", "Arteon", "GTI"],
      "Volvo": ["XC90", "XC60", "XC40", "S60", "S90", "V60", "V90"]
    };
    
    // Years for autocomplete
    const years = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025];
    
    // New function to extract vehicle information from a search query
    function extractVehicleInfo(searchText) {
      // Initialize result object
      const result = {
        year: null,
        make: null,
        model: null,
        product: null
      };
      
      // Convert to lowercase for easier matching
      const searchLower = searchText.toLowerCase();
      
      // Extract year (2-digit or 4-digit)
      const yearMatch = searchLower.match(/\b(19|20)\d{2}\b|\b\d{2}\b/);
      if (yearMatch) {
        result.year = yearMatch[0];
      }
      
      // Extract make and model
      for (const make in vehicleData) {
        const makeLower = make.toLowerCase();
        if (searchLower.includes(makeLower)) {
          result.make = make;
          
          // Try to find model
          const models = vehicleData[make];
          for (const model of models) {
            const modelLower = model.toLowerCase();
            if (searchLower.includes(modelLower)) {
              result.model = model;
              break;
            }
          }
          
          break;
        }
      }
      
      // Extract product by removing year, make, and model
      let product = searchText;
      if (result.year) product = product.replace(result.year, '');
      if (result.make) product = product.replace(new RegExp(result.make, 'i'), '');
      if (result.model) product = product.replace(new RegExp(result.model, 'i'), '');
      
      // Clean up the product text
      result.product = product.replace(/\s+/g, ' ').trim();
      
      return result;
    }
    
    // Initialize autocomplete when the DOM is fully loaded
    document.addEventListener('DOMContentLoaded', function() {
      // Elements for autocomplete
      const searchInput = document.getElementById('search-input');
      const suggestionsContainer = document.getElementById('autocomplete-suggestions');
      
      // Elements for AJAX forms
      const searchForm = document.getElementById('search-form');
      const searchButton = document.getElementById('search-button');
      const resultsContainer = document.getElementById('results-container');
      const validationErrorContainer = document.getElementById('validation-error-container');
      
      const vinForm = document.getElementById('vin-form');
      const vinInput = document.getElementById('vin-input');
      const vinButton = document.getElementById('vin-button');
      const vinResultContainer = document.getElementById('vin-result-container');
      
      // Set up autocomplete
      if (searchInput && suggestionsContainer) {
        // State variables
        let selectedIndex = -1;
        let suggestions = [];
        
        // Listen for input changes
        searchInput.addEventListener('input', function() {
          const query = this.value.toLowerCase().trim();
          
          if (query.length < 2) {
            hideSuggestions();
            return;
          }
          
          // Generate suggestions
          generateSuggestions(query);
        });
        
        // Handle keyboard navigation
        searchInput.addEventListener('keydown', function(e) {
          if (!suggestions.length) return;
          
          if (e.key === 'ArrowDown') {
            e.preventDefault();
            selectedIndex = Math.min(selectedIndex + 1, suggestions.length - 1);
            updateSelectedSuggestion();
          } else if (e.key === 'ArrowUp') {
            e.preventDefault();
            selectedIndex = Math.max(selectedIndex - 1, -1);
            updateSelectedSuggestion();
          } else if (e.key === 'Enter' && selectedIndex >= 0) {
            e.preventDefault();
            selectSuggestion(suggestions[selectedIndex]);
          } else if (e.key === 'Escape') {
            hideSuggestions();
          }
        });
        
        // Close suggestions when clicking outside
        document.addEventListener('click', function(e) {
          if (!suggestionsContainer.contains(e.target) && e.target !== searchInput) {
            hideSuggestions();
          }
        });
        
        // Generate suggestions based on input
        function generateSuggestions(query) {
          suggestions = [];
          
          // Split the query into words and get the last word
          const queryWords = query.split(/\s+/);
          const lastWord = queryWords[queryWords.length - 1].toLowerCase();
          
          // Check if the last word in the query matches a make name
          let lastWordMakeMatch = null;
          for (const make in vehicleData) {
            if (make.toLowerCase() === lastWord) {
              lastWordMakeMatch = make;
              break;
            }
          }
          
          // If the last word is a complete make name, show models for that make
          if (lastWordMakeMatch) {
            // Add all models for this make
            const models = vehicleData[lastWordMakeMatch];
            for (const model of models) {
              suggestions.push(`${lastWordMakeMatch} ${model}`);
            }
          } 
          // Otherwise, perform normal matching
          else {
            // Add matching makes
            for (const make in vehicleData) {
              if (make.toLowerCase().includes(lastWord)) {
                suggestions.push(make);
              }
            }
            
            // Add matching models with their makes
            for (const make in vehicleData) {
              const models = vehicleData[make];
              for (const model of models) {
                if (model.toLowerCase().includes(lastWord)) {
                  suggestions.push(`${make} ${model}`);
                }
              }
            }
            
            // Check for year patterns
            if (lastWord.match(/^(19|20)\d{0,2}$/)) {
              for (const year of years) {
                if (year.toString().startsWith(lastWord)) {
                  suggestions.push(year.toString());
                }
              }
            }
          }
          
          // Add fuzzy matching for typos
          if (suggestions.length === 0 && lastWord.length >= 3) {
            for (let i = 0; i < lastWord.length; i++) {
              const fuzzyWord = lastWord.slice(0, i) + lastWord.slice(i + 1);
              
              for (const make in vehicleData) {
                if (make.toLowerCase().includes(fuzzyWord)) {
                  suggestions.push(make);
                }
              }
              
              if (suggestions.length > 0) break;
            }
          }
          
          // Limit and display suggestions
          suggestions = Array.from(new Set(suggestions)).slice(0, 15);
          
          if (suggestions.length > 0) {
            displaySuggestions();
          } else {
            hideSuggestions();
          }
        }
        
        // Enhanced display function with category labels
        function displaySuggestions() {
          suggestionsContainer.innerHTML = '';
          suggestionsContainer.style.display = 'block';
          
          // Group suggestions by type
          const makes = [];
          const models = [];
          const years = [];
          
          suggestions.forEach(suggestion => {
            if (/^\d{4}$/.test(suggestion)) {
              years.push(suggestion);
            } else if (Object.keys(vehicleData).includes(suggestion)) {
              makes.push(suggestion);
            } else {
              models.push(suggestion);
            }
          });
          
          // Add makes with a header if there are any
          if (makes.length > 0) {
            const makeHeader = document.createElement('div');
            makeHeader.className = 'autocomplete-category';
            makeHeader.textContent = 'MAKES';
            suggestionsContainer.appendChild(makeHeader);
            
            makes.forEach((suggestion, index) => {
              addSuggestionItem(suggestion, index);
            });
          }
          
          // Add models with a header if there are any
          if (models.length > 0) {
            const modelHeader = document.createElement('div');
            modelHeader.className = 'autocomplete-category';
            modelHeader.textContent = 'MODELS';
            suggestionsContainer.appendChild(modelHeader);
            
            models.forEach((suggestion, index) => {
              addSuggestionItem(suggestion, makes.length + index);
            });
          }
          
          // Add years with a header if there are any
          if (years.length > 0) {
            const yearHeader = document.createElement('div');
            yearHeader.className = 'autocomplete-category';
            yearHeader.textContent = 'YEARS';
            suggestionsContainer.appendChild(yearHeader);
            
            years.forEach((suggestion, index) => {
              addSuggestionItem(suggestion, makes.length + models.length + index);
            });
          }
          
          function addSuggestionItem(suggestion, index) {
            const item = document.createElement('div');
            item.className = 'autocomplete-item';
            
            // Highlight the matched part
            const query = searchInput.value.toLowerCase();
            const lastWord = query.split(/\s+/).pop();
            
            if (lastWord && suggestion.toLowerCase().includes(lastWord.toLowerCase())) {
              const highlightIndex = suggestion.toLowerCase().indexOf(lastWord.toLowerCase());
              const beforeMatch = suggestion.substring(0, highlightIndex);
              const match = suggestion.substring(highlightIndex, highlightIndex + lastWord.length);
              const afterMatch = suggestion.substring(highlightIndex + lastWord.length);
              
              item.innerHTML = `${beforeMatch}<strong>${match}</strong>${afterMatch}`;
            } else {
              item.textContent = suggestion;
            }
            
            item.addEventListener('click', () => {
              selectSuggestion(suggestion);
            });
            
            item.addEventListener('mouseover', () => {
              selectedIndex = index;
              updateSelectedSuggestion();
            });
            
            suggestionsContainer.appendChild(item);
          }
        }
        
        // Context-aware select suggestion function
        function selectSuggestion(suggestion) {
          // Check if it's exactly a make name
          const isMake = Object.keys(vehicleData).includes(suggestion);
          
          // Check if it's a model (contains a space with make at the beginning)
          const isModel = suggestion.includes(' ') && Object.keys(vehicleData).some(make => 
            suggestion.startsWith(make + ' ')
          );
          
          // Get the current input value
          const currentValue = searchInput.value;
          
          // Find what we're trying to replace (the last word or the make name)
          const words = currentValue.split(/\s+/);
          const lastWord = words[words.length - 1].toLowerCase();
          
          // Check if the last word is a car make or part of a car make
          let makeToReplace = null;
          for (const make in vehicleData) {
            if (lastWord === make.toLowerCase()) {
              makeToReplace = make;
              break;
            }
          }
          
          // Handle different selection types
          if (isMake) {
            // If the lastWord is part of a search phrase, replace just that word
            if (words.length > 1) {
              words[words.length - 1] = suggestion;
              searchInput.value = words.join(' ');
            } else {
              // Otherwise just set the value to the make
              searchInput.value = suggestion;
            }
            
            // Immediately show model suggestions
            setTimeout(() => {
              generateSuggestions(searchInput.value);
            }, 50);
          } 
          else if (isModel) {
            // Extract the make from the model suggestion
            const makeInModel = Object.keys(vehicleData).find(make => 
              suggestion.startsWith(make + ' ')
            );
            
            if (makeInModel && makeToReplace) {
              // If the last word is a make name, replace it with the full model
              words[words.length - 1] = suggestion;
              searchInput.value = words.join(' ') + ' ';
            } 
            else if (words.length > 1) {
              // Try to find the make in the input to replace it and its related word
              const makeLowerCase = makeInModel.toLowerCase();
              let makeIndex = -1;
              
              for (let i = 0; i < words.length; i++) {
                if (words[i].toLowerCase() === makeLowerCase) {
                  makeIndex = i;
                  break;
                }
              }
              
              if (makeIndex >= 0) {
                // Replace the make and the word after it (if it exists)
                const beforeMake = words.slice(0, makeIndex).join(' ');
                const afterModel = words.slice(makeIndex + 2).join(' ');
                
                if (beforeMake && afterModel) {
                  searchInput.value = beforeMake + ' ' + suggestion + ' ' + afterModel;
                } else if (beforeMake) {
                  searchInput.value = beforeMake + ' ' + suggestion + ' ';
                } else if (afterModel) {
                  searchInput.value = suggestion + ' ' + afterModel;
                } else {
                  searchInput.value = suggestion + ' ';
                }
              } else {
                // If we can't find the make, just replace the last word
                words[words.length - 1] = suggestion;
                searchInput.value = words.join(' ') + ' ';
              }
            } else {
              // Simple case - just replace the entire input
              searchInput.value = suggestion + ' ';
            }
            
            hideSuggestions();
          }
          else {
            // For other suggestions (years, parts, etc.)
            if (words.length > 1) {
              // Replace the last word
              words[words.length - 1] = suggestion;
              searchInput.value = words.join(' ') + ' ';
            } else {
              // Just set the value and add a space
              searchInput.value = suggestion + ' ';
            }
            
            hideSuggestions();
          }
          
          searchInput.focus();
        }
        
        // Update the selected suggestion in UI
        function updateSelectedSuggestion() {
          const items = suggestionsContainer.querySelectorAll('.autocomplete-item');
          items.forEach((item, index) => {
            if (index === selectedIndex) {
              item.classList.add('selected');
            } else {
              item.classList.remove('selected');
            }
          });
        }
        
        // Hide suggestions
        function hideSuggestions() {
          suggestionsContainer.style.display = 'none';
          suggestionsContainer.innerHTML = '';
          selectedIndex = -1;
        }
      }
      
      // AJAX for Search form
      if (searchForm && searchButton && resultsContainer) {
        const searchProgress = document.getElementById('search-progress');
        
        searchForm.addEventListener('submit', function(e) {
          e.preventDefault();
          
          // Get form data
          const formData = new FormData(searchForm);
          const query = formData.get('prompt').trim();
          
          if (!query) {
            validationErrorContainer.innerHTML = `
              <div class="error-box mt-3 fade-in">
                <p class="mb-0"><strong>⚠️ Error:</strong> Please enter a search query</p>
              </div>
            `;
            return;
          }
          
          // Update button state
          searchButton.disabled = true;
          searchButton.innerHTML = '<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span> Thinking...';
          searchButton.classList.add('opacity-75');
          
          // Clear previous results
          validationErrorContainer.innerHTML = '';
          resultsContainer.innerHTML = '';
          
          // Show progress indicator
          if (searchProgress) {
            searchProgress.style.display = 'block';
          }
          
          // Set a timeout to handle slow requests
          const timeoutId = setTimeout(() => {
            if (searchProgress) {
              const progressText = searchProgress.querySelector('p');
              if (progressText) {
                progressText.textContent = 'This is taking longer than expected. Still searching...';
              }
            }
          }, 10000); // 10 seconds
          
          // Submit the form via AJAX
          fetch('/api/search', {
            method: 'POST',
            body: formData,
            // Add a timeout of 60 seconds
            signal: AbortSignal.timeout(60000)
          })
          .then(response => {
            if (!response.ok) {
              throw new Error(`Server responded with status: ${response.status}`);
            }
            return response.json();
          })
          .then(data => {
            // Clear timeout
            clearTimeout(timeoutId);
            
            // Reset button state
            searchButton.disabled = false;
            searchButton.innerHTML = '🔍 Search';
            searchButton.classList.remove('opacity-75');
            
            // Hide progress indicator
            if (searchProgress) {
              searchProgress.style.display = 'none';
            }
            
            // Handle validation errors
            if (!data.success && data.validation_error) {
              validationErrorContainer.innerHTML = `
                <div class="error-box mt-3 fade-in">
                  <p class="mb-0"><strong>⚠️ Missing vehicle information:</strong> ${data.validation_error}</p>
                </div>
              `;
              return;
            }
            
            // Handle general errors
            if (!data.success) {
              validationErrorContainer.innerHTML = `
                <div class="error-box mt-3 fade-in">
                  <p class="mb-0"><strong>⚠️ Error:</strong> ${data.error || 'An unknown error occurred'}</p>
                  <p class="mb-0 mt-2"><small>Please try again or refine your search.</small></p>
                </div>
              `;
              return;
            }
            
            // Build the content for the questions
            if (data.questions) {
              let questionsHtml = `
                <hr class="my-4">
                <h5 class="mb-3 section-title">🧠 Ask the customer:</h5>
                <div class="highlight-box fade-in">
              `;
              
              // Split the questions and format them
              const lines = data.questions.split('\n');
              for (const line of lines) {
                if (line.trim()) {
                  questionsHtml += `<p class="mb-2">👉 ${line}</p>`;
                }
              }
              
              // Add the Schedule Callback button
              questionsHtml += `
                </div>
                <button id="schedule-callback" class="btn-callback fade-in">
                  <i> 📅 </i>  Schedule Callback
                </button>
              `;
              
              resultsContainer.innerHTML += questionsHtml;
              
              setTimeout(function() {
                const callbackButton = document.getElementById('schedule-callback');
                if (callbackButton) {
                  callbackButton.addEventListener('click', function() {
                    const searchQuery = document.getElementById('search-input').value;
                    
                    // Extract vehicle info using the existing vehicleData object
                    const searchLower = searchQuery.toLowerCase();
                    let vehicleInfo = '';
                    let year = '';
                    
                    // Extract year
                    const yearMatch = searchQuery.match(/\b(19|20)\d{2}\b/);
                    if (yearMatch) {
                      year = yearMatch[0];
                    }
                    
                    // Extract make and model
                    for (const make in vehicleData) {
                      if (searchLower.includes(make.toLowerCase())) {
                        vehicleInfo = make;
                        
                        // Try to find model
                        const models = vehicleData[make];
                        for (const model of models) {
                          if (searchLower.includes(model.toLowerCase())) {
                            vehicleInfo += ' ' + model;
                            break;
                          }
                        }
                        break;
                      }
                    }
                    
                    // Extract product by removing year and vehicle info
                    let product = searchQuery;
                    if (year) product = product.replace(year, '');
                    if (vehicleInfo) product = product.replace(vehicleInfo, '');
                    product = product.trim();
                    
                    // Redirect with query parameters
                    window.location.href = `https://autoxpress.us/callbacks/?vehicle=${encodeURIComponent(vehicleInfo)}&year=${encodeURIComponent(year)}&product=${encodeURIComponent(product)}`;
                  });
                }
              }, 100);
            }
            
                 // Build the content for listings
                 if (data.listings && data.listings.length > 0) {
                // Calculate price range
                const prices = data.listings
                  .map(item => item.price)
                  .filter(price => price)
                  .map(price => parseFloat(price.replace('$', '').replace(',', '')))
                  .filter(price => !isNaN(price));
                
                let minPrice = 0;
                let maxPrice = 0;
                
                if (prices.length > 0) {
                  minPrice = Math.min(...prices);
                  maxPrice = Math.max(...prices);
                }
                
                let listingsHtml = `
                  <hr class="my-4">
                  <h5 class="mb-3 section-title">💰 Product Listings:</h5>
                `;
                
                if (prices.length > 0) {
                  listingsHtml += `
                    <p class="price-box"><strong>💲 Price Range:</strong> $${minPrice.toFixed(2)} – $${maxPrice.toFixed(2)}</p>
                  `;
                }
                
                listingsHtml += `<ul class="list-group">`;

                for (const item of data.listings) {
                  listingsHtml += `
                    <li class="list-group-item d-flex justify-content-between align-items-start">
                      <div class="me-auto">
                        <strong>${item.title}</strong><br>
                        <div class="d-flex align-items-center mb-1">
                          <span class="price-box">${item.price}</span>
                          <span class="badge bg-secondary ms-2 shipping-tag">${item.shipping || 'Shipping not specified'}</span>
                        </div>
                        <span class="badge ${item.condition && item.condition.toLowerCase().includes('new') ? 'bg-success' : 'bg-warning text-dark'} mb-2">${item.condition || 'Condition not specified'}</span><br>
                        <a href="${item.link}" target="_blank" class="btn btn-sm btn-outline-primary mt-1">🔗 View Product</a>
                      </div>
                    </li>
                  `;
                }

                listingsHtml += `</ul>`;
                
                listingsHtml += `</ul>`;
                resultsContainer.innerHTML += listingsHtml;
              } else if (data.success) {
                resultsContainer.innerHTML += `
                  <div class="text-danger mt-3"><strong>⚠️ No listings returned.</strong></div>
                `;
              }
          });
        });
      }
      
      // AJAX for VIN form
      if (vinForm && vinButton && vinResultContainer) {
        const vinProgress = document.getElementById('vin-progress');
        
        vinForm.addEventListener('submit', function(e) {
          e.preventDefault();
          
          // Get the VIN value
          const vin = vinInput.value.trim();
          
          if (!vin) {
            vinResultContainer.innerHTML = `
              <div class="alert alert-danger mt-3 fade-in">
                <strong>Error:</strong> Please enter a VIN
              </div>
            `;
            return;
          }
          
          // Validate VIN format (17 characters, no I, O, Q)
          if (!/^[A-HJ-NPR-Z0-9]{17}$/.test(vin)) {
            vinResultContainer.innerHTML = `
              <div class="alert alert-danger mt-3 fade-in">
                <strong>Error:</strong> Invalid VIN format. VIN should be 17 alphanumeric characters (excluding I, O, and Q).
              </div>
            `;
            return;
          }
          
          // Update button state
          vinButton.disabled = true;
          vinButton.innerHTML = '<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span> Decoding...';
          vinButton.classList.add("opacity-75");
          
          // Clear previous results and show progress
          vinResultContainer.innerHTML = '';
          if (vinProgress) {
            vinProgress.style.display = 'block';
          }
          
          // Set a timeout to handle slow requests
          const timeoutId = setTimeout(() => {
            if (vinProgress) {
              const progressText = vinProgress.querySelector('p');
              if (progressText) {
                progressText.textContent = 'This is taking longer than expected. Still decoding...';
              }
            }
          }, 5000); // 5 seconds
          
          // Make an AJAX request to decode the VIN
          fetch('/api/vin-decode', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: `vin=${encodeURIComponent(vin)}`,
            // Add a timeout of 20 seconds
            signal: AbortSignal.timeout(20000)
          })
          .then(response => {
            if (!response.ok) {
              throw new Error(`Server responded with status: ${response.status}`);
            }
            return response.json();
          })
          .then(data => {
            // Clear timeout and hide progress
            clearTimeout(timeoutId);
            if (vinProgress) {
              vinProgress.style.display = 'none';
            }
            
            // Reset button state
            vinButton.disabled = false;
            vinButton.innerHTML = 'Decode VIN';
            vinButton.classList.remove("opacity-75");
            
            // Display the result
            if (data.error) {
              vinResultContainer.innerHTML = `
                <div class="alert alert-danger mt-3 fade-in">
                  <strong>Error:</strong> ${data.error}
                </div>
              `;
            } else {
              vinResultContainer.innerHTML = `
                <div class="card shadow mt-4 fade-in">
                  <div class="card-body">
                    <h5 class="card-title">🔍 Decoded VIN Information</h5>
                    <ul class="list-group list-group-flush">
                      <li class="list-group-item"><strong>Make:</strong> ${data.Make || 'N/A'}</li>
                      <li class="list-group-item"><strong>Model:</strong> ${data.Model || 'N/A'}</li>
                      <li class="list-group-item"><strong>Year:</strong> ${data.ModelYear || 'N/A'}</li>
                      <li class="list-group-item"><strong>Trim:</strong> ${data.Trim || 'N/A'}</li>
                      <li class="list-group-item"><strong>Engine:</strong> ${data.EngineModel || 'N/A'} ${data.EngineCylinders ? `(${data.EngineCylinders} cylinders)` : ''}</li>
                      <li class="list-group-item"><strong>Drivetrain:</strong> ${data.DriveType || 'N/A'}</li>
                    </ul>
                    <div class="mt-3">
                      <button class="btn btn-sm btn-primary copy-to-search">Use this vehicle</button>
                    </div>
                  </div>
                </div>
              `;
              
              // Add event listener for "Use this vehicle" button
              const copyButton = vinResultContainer.querySelector('.copy-to-search');
              if (copyButton) {
                copyButton.addEventListener('click', function() {
                  const vehicleInfo = `${data.ModelYear} ${data.Make} ${data.Model} ${data.Trim || ''}`;
                  const searchInput = document.getElementById('search-input');
                  if (searchInput) {
                    searchInput.value = vehicleInfo.trim() + ' ';
                    searchInput.focus();
                  }
                });
              }
            }
          })
          .catch(error => {
            // Clear timeout and hide progress
            clearTimeout(timeoutId);
            if (vinProgress) {
              vinProgress.style.display = 'none';
            }
            
            console.error('Error:', error);
            vinButton.disabled = false;
            vinButton.innerHTML = 'Decode VIN';
            vinButton.classList.remove("opacity-75");
            
            vinResultContainer.innerHTML = `
              <div class="alert alert-danger mt-3 fade-in">
                <strong>Error:</strong> ${error.message || 'Failed to decode VIN. Please try again.'}
              </div>
            `;
          });
        });
      }
    });
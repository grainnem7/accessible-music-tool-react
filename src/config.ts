// config.ts
// Azure Configuration for accessible music creation tool

export const azureConfig = {
    // Computer Vision configuration
    computerVisionEndpoint: 'https://reactmusicvision.cognitiveservices.azure.com/',
    computerVisionKey1: '9ZWwcipd56k45lgE5KYEGMJyGQ5HgfrcDbjaJHta1L4C0Bn671BwJQQJ99BCACYeBjFXJ3w3AAAFACOGlFcE',
    computerVisionKey2: '43nfYXDI17ZvFFsSA88zpH9ohdtwRfLOQDU5TaHLpp0bJa5fFPNUJQQJ99BCACYeBjFXJ3w3AAAFACOGzPB5',
    computerVisionLocation: 'eastus',
    
    // Azure Custom Vision configuration
    customVisionEndpoint: 'https://eastus.api.cognitive.microsoft.com/',
    customVisionKey: '1234abcd5678efgh', // Replace with your actual key
    customVisionTrainingKey: '5678efgh1234abcd', // Replace with your actual training key
    customVisionProjectId: 'project-guid-here', // Replace with your project ID
    customVisionModelName: 'IntentionalMovements-1.0', // Replace with your model iteration name
    intentionalTagId: 'intentional-tag-guid', // Replace with your tag ID
    unintentionalTagId: 'unintentional-tag-guid', // Replace with your tag ID
    
    // Azure Speech Service configuration
    speechKey: 'speech-key-here', // Replace with your speech service key
    speechRegion: 'eastus', // Replace with your speech service region
    
    // Azure ML configuration
    mlEndpoint: 'https://your-ml-endpoint.azureml.net/score', // Replace with your endpoint
    mlApiKey: 'ml-api-key-here', // Replace with your ML API key
    mlTrainingEndpoint: 'https://your-ml-training-endpoint', // Replace with your ML training endpoint
    
    // Use the first Computer Vision key by default
    get computerVisionKey() {
      return this.computerVisionKey1;
    }
  };
  
  // Function to create Azure API headers
  export const getAzureHeaders = () => {
    return {
      'Ocp-Apim-Subscription-Key': azureConfig.computerVisionKey,
      'Content-Type': 'application/octet-stream'
    };
  };
  
  // Function to build Azure API URL with parameters
  export const buildAzureApiUrl = (endpoint: string, params: Record<string, string>) => {
    const url = new URL(endpoint, azureConfig.computerVisionEndpoint);
    
    Object.entries(params).forEach(([key, value]) => {
      url.searchParams.append(key, value);
    });
    
    return url.toString();
  };
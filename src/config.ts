// config.ts
// Azure Computer Vision configuration

export const azureConfig = {
    computerVisionEndpoint: 'https://reactmusicvision.cognitiveservices.azure.com/',
    computerVisionKey1: '9ZWwcipd56k45lgE5KYEGMJyGQ5HgfrcDbjaJHta1L4C0Bn671BwJQQJ99BCACYeBjFXJ3w3AAAFACOGlFcE',
    computerVisionKey2: '43nfYXDI17ZvFFsSA88zpH9ohdtwRfLOQDU5TaHLpp0bJa5fFPNUJQQJ99BCACYeBjFXJ3w3AAAFACOGzPB5',
    computerVisionLocation: 'eastus',
    
    // Use the first key by default
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
import { PUBLIC_API_URL } from '$env/static/public';

export interface PredictionResult {
    class_name: string;
    confidence: number;
    inference_time_ms: number;
}

export interface ApiError {
    detail: string;
}

/**
 * Upload an image to the API and get a prediction
 */
export async function predictImage(file: File): Promise<PredictionResult> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${PUBLIC_API_URL}/v1/predict`, {
        method: 'POST',
        body: formData
    });

    if (!response.ok) {
        const error: ApiError = await response.json();
        throw new Error(error.detail || 'Prediction failed');
    }

    const data = await response.json();

    return {
        class_name: data.prediction.class_name,
        confidence: data.prediction.confidence,
        inference_time_ms: data.metadata.inference_time_ms
    };
}

/**
 * Check API health
 */
export async function checkHealth(): Promise<{ status: string; model_loaded: boolean }> {
    const response = await fetch(`${PUBLIC_API_URL}/health`);
    return response.json();
}

/**
 * VeriFace Frontend Configuration & API Client Helper
 * Dynamically resolves the API base URL from Vite environment variables (VITE_API_BASE_URL)
 * or defaults to relative endpoints for reverse proxy/local setups.
 */

export const API_BASE = import.meta.env.VITE_API_BASE_URL || '';

export const getApiUrl = (path = '') => {
  if (!path) return '';
  if (
    path.startsWith('http://') ||
    path.startsWith('https://') ||
    path.startsWith('data:') ||
    path.startsWith('blob:')
  ) {
    return path;
  }
  const cleanPath = path.startsWith('/') ? path : `/${path}`;
  return `${API_BASE}${cleanPath}`;
};


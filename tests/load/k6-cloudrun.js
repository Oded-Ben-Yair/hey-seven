import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend, Counter } from 'k6/metrics';

// Custom metrics
const errorRate = new Rate('errors');
const firstTokenTime = new Trend('first_token_time', true);
const scaleUpErrors = new Counter('scale_up_503s');

// Required environment variables
const BASE_URL = __ENV.CLOUD_RUN_URL;
const API_KEY = __ENV.API_KEY;

if (!BASE_URL) {
  throw new Error('CLOUD_RUN_URL environment variable is required');
}
if (!API_KEY) {
  throw new Error('API_KEY environment variable is required');
}

export const options = {
  stages: [
    { duration: '30s', target: 25 },   // Ramp up
    { duration: '1m', target: 50 },     // Moderate load
    { duration: '2m', target: 100 },    // Peak: 100 concurrent SSE streams
    { duration: '30s', target: 50 },    // Ramp down
    { duration: '30s', target: 0 },     // Drain
  ],
  thresholds: {
    // Cloud Run thresholds are higher than local (network latency + cold starts)
    'http_req_duration': ['p(95)<3000', 'p(99)<8000'],
    'errors': ['rate<0.01'],                              // Error rate < 1%
    'first_token_time': ['p(95)<3000'],                   // First token p95 < 3s
    'scale_up_503s': ['count<10'],                        // Max 10 503s during autoscale
  },
};

// Casino-specific chat queries
const QUERIES = [
  'What restaurants are open tonight?',
  'Tell me about the spa services',
  'What shows are playing this weekend?',
  'How do I earn comp dollars?',
  'What are the hotel room rates?',
];

export default function () {
  const headers = {
    'Content-Type': 'application/json',
    'X-API-Key': API_KEY,
  };

  const query = QUERIES[Math.floor(Math.random() * QUERIES.length)];

  const startTime = Date.now();
  const res = http.post(`${BASE_URL}/chat`, JSON.stringify({
    message: query,
    thread_id: `k6-cloud-${__VU}-${__ITER}`,
  }), { headers, timeout: '30s' });

  const elapsed = Date.now() - startTime;
  firstTokenTime.add(elapsed);

  // Track 503s separately (expected during Cloud Run autoscaling)
  if (res.status === 503) {
    scaleUpErrors.add(1);
  }

  check(res, {
    'chat: status 200': (r) => r.status === 200,
    'chat: has body': (r) => r.body && r.body.length > 0,
    'chat: not rate limited': (r) => r.status !== 429,
    'chat: no server error': (r) => r.status < 500,
  });

  // Only count 5xx as errors (429 is expected under load)
  errorRate.add(res.status >= 500);

  // Think time: 1-3s between requests
  sleep(1 + Math.random() * 2);
}

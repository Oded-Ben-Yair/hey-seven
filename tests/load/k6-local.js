import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend } from 'k6/metrics';

// Custom metrics
const errorRate = new Rate('errors');
const firstTokenTime = new Trend('first_token_time', true);

// Test configuration
const BASE_URL = __ENV.BASE_URL || 'http://localhost:8080';
const API_KEY = __ENV.API_KEY || '';

export const options = {
  stages: [
    { duration: '30s', target: 25 },   // Ramp up to 25 VUs
    { duration: '1m', target: 50 },     // Ramp to 50
    { duration: '2m', target: 100 },    // Hold at 100 concurrent
    { duration: '30s', target: 50 },    // Ramp down
    { duration: '30s', target: 0 },     // Drain
  ],
  thresholds: {
    'http_req_duration': ['p(95)<2000', 'p(99)<5000'],  // p95 < 2s, p99 < 5s
    'errors': ['rate<0.01'],                              // Error rate < 1%
    'first_token_time': ['p(95)<2000'],                   // First token p95 < 2s
  },
};

// Casino-specific chat queries for realistic load distribution
const QUERIES = [
  'What restaurants are open tonight?',
  'Tell me about the spa services',
  'What shows are playing this weekend?',
  'How do I earn comp dollars?',
  'What are the hotel room rates?',
];

// Health check scenario (used for warm-up)
export function healthCheck() {
  const res = http.get(`${BASE_URL}/health`);
  check(res, {
    'health: status 200': (r) => r.status === 200,
    'health: agent_ready': (r) => {
      try { return JSON.parse(r.body).agent_ready === true; }
      catch { return false; }
    },
  });
  errorRate.add(res.status !== 200);
}

// SSE chat scenario (default function)
export default function () {
  const headers = { 'Content-Type': 'application/json' };
  if (API_KEY) {
    headers['X-API-Key'] = API_KEY;
  }

  const query = QUERIES[Math.floor(Math.random() * QUERIES.length)];

  const startTime = Date.now();
  const res = http.post(`${BASE_URL}/chat`, JSON.stringify({
    message: query,
    thread_id: `k6-local-${__VU}-${__ITER}`,
  }), { headers, timeout: '30s' });

  const elapsed = Date.now() - startTime;
  firstTokenTime.add(elapsed);

  check(res, {
    'chat: status 200': (r) => r.status === 200,
    'chat: has body': (r) => r.body && r.body.length > 0,
    'chat: not rate limited': (r) => r.status !== 429,
  });

  // Only count 5xx as errors (429 is expected under load)
  errorRate.add(res.status >= 500);

  // Think time: 1-3s between requests (simulates real user behavior)
  sleep(1 + Math.random() * 2);
}

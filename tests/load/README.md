# Load Tests (k6)

## Prerequisites

Install k6: https://grafana.com/docs/k6/latest/set-up/install-k6/

## Local (docker-compose)

```bash
# Start the app
make docker-up

# Run load test (100 concurrent users, 5 minutes)
make load-test-local

# Or with custom settings
k6 run tests/load/k6-local.js --env BASE_URL=http://localhost:8080
```

## Cloud Run

```bash
# Run against Cloud Run (requires API key)
make load-test-cloud

# Or with custom settings
k6 run tests/load/k6-cloudrun.js \
  --env CLOUD_RUN_URL=https://hey-seven-XXXXX.run.app \
  --env API_KEY=your-api-key
```

## Thresholds

| Metric | Local | Cloud Run |
|--------|-------|-----------|
| Response p95 | < 2s | < 3s |
| Response p99 | < 5s | < 8s |
| First token p95 | < 2s | < 3s |
| Error rate | < 1% | < 1% |
| Scale-up 503s | N/A | < 10 |

## Interpreting Results

- **errors**: Percentage of 5xx responses (target: < 1%). 429s are expected and excluded.
- **first_token_time**: Time to first SSE response token.
- **http_req_duration**: Full request-response time.
- **scale_up_503s**: 503s during Cloud Run autoscaling (expected during ramp-up, but < 10).

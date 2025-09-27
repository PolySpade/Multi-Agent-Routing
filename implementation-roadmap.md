# MAS-FRO Implementation Roadmap
## From Foundation to Production: Action Plan

**Generated:** September 2025  
**Priority:** High  
**Timeline:** 10 weeks to production

---

## 🎯 Current State Analysis

### What's Working Well ✅
1. **RoutingAgent Core Algorithm** - The risk-aware A* implementation is solid and research-based
2. **Agent Architecture** - Clean separation of concerns with BaseAgent pattern
3. **Data Structures** - Well-defined models for all data types
4. **Basic Communication** - Queue-based messaging between agents works
5. **Documentation** - Comprehensive docstrings and clear code structure

### Critical Gaps to Address 🚨
1. **No Real Data Sources** - All data is simulated/mocked
2. **No User Interface** - No way for users to interact with the system
3. **No Persistence** - System loses all state on restart
4. **Limited Scalability** - Single-instance, single-threaded design
5. **No Monitoring** - Can't track system health or performance

---

## 📋 Week 1-2: Data Integration & Reliability

### Objective
Transform the system from mock data to real data sources while ensuring reliability.

### Concrete Tasks

#### 1.1 Create Data Integration Service
```python
# src/services/data_integration.py
import aiohttp
import asyncio
from typing import List, Optional
from datetime import datetime
import json

class PAGASAClient:
    """Client for PAGASA weather API"""
    #not real url - so far
    BASE_URL = "https://api.pagasa.gov.ph/v1"
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            headers={'Authorization': f'Bearer {self.api_key}'}
        )
        return self
    
    async def __aexit__(self, *args):
        if self.session:
            await self.session.close()
    
    async def get_water_levels(self) -> List[dict]:
        """Fetch current water levels from monitoring stations"""
        async with self.session.get(
            f"{self.BASE_URL}/water-levels",
            params={'region': 'NCR', 'include_forecast': 'true'}
        ) as response:
            if response.status == 200:
                data = await response.json()
                return self._transform_water_data(data)
            else:
                raise APIError(f"PAGASA API error: {response.status}")
    
    def _transform_water_data(self, raw_data: dict) -> List[dict]:
        """Transform PAGASA data to our FloodData format"""
        transformed = []
        for station in raw_data.get('stations', []):
            transformed.append({
                'station_id': station['id'],
                'water_level': station['current_level'],
                'rainfall_intensity': station.get('rainfall', 0),
                'location': (station['lat'], station['lon']),
                'timestamp': datetime.fromisoformat(station['timestamp']),
                'alert_level': station.get('alert_level', 'normal')
            })
        return transformed
```

#### 1.2 Add Data Validation
```python
# src/services/data_validator.py
from pydantic import BaseModel, validator, Field
from typing import Tuple, Optional
from datetime import datetime

class ValidatedFloodData(BaseModel):
    """Validated flood data with automatic type checking"""
    
    station_id: str = Field(..., min_length=3, max_length=20)
    water_level: float = Field(..., ge=0, le=50)  # 0-50 meters
    rainfall_intensity: float = Field(..., ge=0, le=200)  # 0-200 mm/hr
    location: Tuple[float, float]
    timestamp: datetime
    alert_level: str = Field(..., regex='^(normal|warning|critical|danger)$')
    
    @validator('location')
    def validate_philippines_coordinates(cls, v):
        lat, lon = v
        # Validate within Philippines bounds
        if not (4.5 <= lat <= 21.5 and 116.0 <= lon <= 127.0):
            raise ValueError('Coordinates outside Philippines')
        return v
    
    @validator('water_level')
    def validate_realistic_water_level(cls, v, values):
        if 'alert_level' in values:
            if values['alert_level'] == 'danger' and v < 10:
                raise ValueError('Danger level requires water_level >= 10m')
        return v
```

#### 1.3 Implement Caching Layer
```python
# src/services/cache_manager.py
import redis
import pickle
import hashlib
from typing import Any, Optional
from datetime import timedelta

class CacheManager:
    """Centralized caching service with TTL support"""
    
    def __init__(self, redis_url: str = 'redis://localhost:6379'):
        self.redis_client = redis.from_url(redis_url)
        self.default_ttl = 300  # 5 minutes
    
    def _generate_key(self, prefix: str, params: dict) -> str:
        """Generate cache key from parameters"""
        param_str = json.dumps(params, sort_keys=True)
        hash_digest = hashlib.md5(param_str.encode()).hexdigest()
        return f"{prefix}:{hash_digest}"
    
    def get_route(self, origin: Tuple, destination: str) -> Optional[dict]:
        """Get cached route if available"""
        key = self._generate_key('route', {
            'origin': origin,
            'destination': destination
        })
        
        cached = self.redis_client.get(key)
        if cached:
            return pickle.loads(cached)
        return None
    
    def set_route(self, origin: Tuple, destination: str, 
                  route: dict, ttl: int = None):
        """Cache route with TTL"""
        key = self._generate_key('route', {
            'origin': origin,
            'destination': destination
        })
        
        self.redis_client.set(
            key, 
            pickle.dumps(route),
            ex=ttl or self.default_ttl
        )
    
    def invalidate_area(self, center: Tuple, radius_km: float):
        """Invalidate all cached routes in affected area"""
        # Implementation: scan and delete affected routes
        pattern = 'route:*'
        for key in self.redis_client.scan_iter(pattern):
            route = pickle.loads(self.redis_client.get(key))
            if self._route_intersects_area(route, center, radius_km):
                self.redis_client.delete(key)
```

#### 1.4 Add Resilience Patterns
```python
# src/utils/resilience.py
import asyncio
from functools import wraps
from typing import Callable, Any
import logging

logger = logging.getLogger(__name__)

class CircuitBreaker:
    """Circuit breaker pattern for external service calls"""
    
    def __init__(self, failure_threshold: int = 5, 
                 recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
    
    def __call__(self, func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if self.state == 'OPEN':
                if self._should_attempt_reset():
                    self.state = 'HALF_OPEN'
                else:
                    raise CircuitBreakerOpen(f"{func.__name__} circuit open")
            
            try:
                result = await func(*args, **kwargs)
                self._on_success()
                return result
            except Exception as e:
                self._on_failure()
                raise
        
        return wrapper
    
    def _on_success(self):
        self.failure_count = 0
        self.state = 'CLOSED'
    
    def _on_failure(self):
        self.failure_count += 1
        self.last_failure_time = asyncio.get_event_loop().time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = 'OPEN'
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
    
    def _should_attempt_reset(self) -> bool:
        return (
            asyncio.get_event_loop().time() - self.last_failure_time 
            >= self.recovery_timeout
        )

def retry_with_backoff(max_retries: int = 3, 
                       initial_delay: float = 1.0,
                       exponential_base: float = 2.0):
    """Decorator for retry with exponential backoff"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            delay = initial_delay
            
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s")
                    await asyncio.sleep(delay)
                    delay *= exponential_base
        
        return wrapper
    return decorator
```

### Deliverables for Week 1-2
- [ ] Working PAGASA API integration
- [ ] Data validation for all inputs
- [ ] Redis caching implemented
- [ ] Circuit breaker pattern for external services
- [ ] Retry mechanisms with exponential backoff
- [ ] Integration tests for data pipeline

---

## 📋 Week 3-4: API & Basic UI

### Objective
Create REST API and basic web interface for user interaction.

### Concrete Tasks

#### 2.1 Build FastAPI Backend
```python
# src/api/main.py
from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
from typing import List, Optional

app = FastAPI(title="MAS-FRO API", version="1.0.0")

# Add CORS for web frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class RouteRequestModel(BaseModel):
    origin: Tuple[float, float]
    destination: str
    user_id: str
    priority: str = "normal"

@app.post("/api/v1/routes/calculate")
async def calculate_route(request: RouteRequestModel):
    """Calculate optimal evacuation route"""
    try:
        # Send to routing agent
        route = await route_service.calculate(
            origin=request.origin,
            destination=request.destination,
            user_id=request.user_id
        )
        
        return {
            "success": True,
            "route": route,
            "cached": False
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/updates")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time updates"""
    await websocket.accept()
    
    try:
        while True:
            # Send updates every second
            update = await update_queue.get()
            await websocket.send_json(update)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await websocket.close()

@app.get("/api/v1/status")
async def system_status():
    """Get system status and agent health"""
    return {
        "agents": agent_monitor.get_status(),
        "performance": performance_monitor.get_metrics(),
        "alerts": alert_manager.get_active_alerts()
    }
```

#### 2.2 Create React Dashboard
```typescript
// web/src/pages/Dashboard.tsx
import React, { useState, useEffect } from 'react';
import { MapContainer, TileLayer, Polyline, Marker } from 'react-leaflet';
import { useWebSocket } from '@/hooks/useWebSocket';
import { RoutePanel } from '@/components/RoutePanel';
import { AlertBanner } from '@/components/AlertBanner';

export const Dashboard: React.FC = () => {
    const [route, setRoute] = useState(null);
    const [hazards, setHazards] = useState([]);
    const { messages, sendMessage } = useWebSocket('ws://localhost:8000/ws/updates');
    
    const calculateRoute = async () => {
        const response = await fetch('/api/v1/routes/calculate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                origin: [14.6507, 121.1029],
                destination: 'Marikina Sports Center',
                user_id: 'web_user_001'
            })
        });
        
        const data = await response.json();
        if (data.success) {
            setRoute(data.route);
        }
    };
    
    useEffect(() => {
        // Handle WebSocket updates
        messages.forEach(msg => {
            if (msg.type === 'hazard_update') {
                setHazards(msg.data);
            }
        });
    }, [messages]);
    
    return (
        <div className="dashboard">
            <AlertBanner alerts={hazards.filter(h => h.risk_level > 0.7)} />
            
            <div className="main-content">
                <MapContainer 
                    center={[14.6507, 121.1029]} 
                    zoom={13}
                    className="map-container"
                >
                    <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
                    
                    {route && (
                        <Polyline 
                            positions={route.coordinates}
                            color={getRouteColor(route.safety_score)}
                            weight={4}
                        />
                    )}
                    
                    {hazards.map(hazard => (
                        <Marker 
                            key={hazard.id}
                            position={hazard.location}
                            icon={getHazardIcon(hazard.risk_level)}
                        />
                    ))}
                </MapContainer>
                
                <RoutePanel 
                    route={route}
                    onCalculateRoute={calculateRoute}
                />
            </div>
        </div>
    );
};
```

### Deliverables for Week 3-4
- [ ] FastAPI backend with all endpoints
- [ ] WebSocket real-time updates
- [ ] React dashboard with map
- [ ] Route visualization
- [ ] Real-time hazard display
- [ ] Basic mobile-responsive design

---

## 📋 Week 5-6: Message Broker & Scalability

### Objective
Replace queue-based messaging with proper message broker for scalability.

### Concrete Tasks

#### 3.1 Implement RabbitMQ Integration
```python
# src/messaging/rabbitmq_service.py
import pika
import json
import asyncio
from typing import Callable

class RabbitMQService:
    """RabbitMQ message broker service"""
    
    def __init__(self, connection_url: str):
        self.connection_url = connection_url
        self.connection = None
        self.channel = None
        self.setup_complete = False
    
    async def connect(self):
        """Establish connection and declare exchanges/queues"""
        parameters = pika.URLParameters(self.connection_url)
        self.connection = pika.BlockingConnection(parameters)
        self.channel = self.connection.channel()
        
        # Declare exchanges
        self.channel.exchange_declare(
            exchange='mas_fro',
            exchange_type='topic',
            durable=True
        )
        
        # Declare queues with dead letter exchange
        queues = [
            'routing_requests',
            'hazard_updates',
            'flood_data',
            'scout_reports'
        ]
        
        for queue in queues:
            self.channel.queue_declare(
                queue=queue,
                durable=True,
                arguments={
                    'x-dead-letter-exchange': 'mas_fro_dlx',
                    'x-message-ttl': 3600000  # 1 hour
                }
            )
            
            # Bind queues to exchange
            routing_key = queue.replace('_', '.')
            self.channel.queue_bind(
                queue=queue,
                exchange='mas_fro',
                routing_key=routing_key
            )
        
        self.setup_complete = True
    
    async def publish(self, routing_key: str, message: dict):
        """Publish message to exchange"""
        if not self.setup_complete:
            await self.connect()
        
        self.channel.basic_publish(
            exchange='mas_fro',
            routing_key=routing_key,
            body=json.dumps(message),
            properties=pika.BasicProperties(
                delivery_mode=2,  # Persistent
                content_type='application/json'
            )
        )
    
    def subscribe(self, queue: str, callback: Callable):
        """Subscribe to queue with callback"""
        def wrapper(channel, method, properties, body):
            message = json.loads(body)
            try:
                callback(message)
                channel.basic_ack(delivery_tag=method.delivery_tag)
            except Exception as e:
                # Requeue on error
                channel.basic_nack(
                    delivery_tag=method.delivery_tag,
                    requeue=True
                )
        
        self.channel.basic_consume(
            queue=queue,
            on_message_callback=wrapper,
            auto_ack=False
        )
```

#### 3.2 Add Container Orchestration
```yaml
# docker-compose.yml
version: '3.9'

services:
  rabbitmq:
    image: rabbitmq:3-management
    ports:
      - "5672:5672"
      - "15672:15672"
    environment:
      RABBITMQ_DEFAULT_USER: admin
      RABBITMQ_DEFAULT_PASS: admin
    volumes:
      - rabbitmq_data:/var/lib/rabbitmq

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  postgres:
    image: postgis/postgis:15-3.3
    environment:
      POSTGRES_DB: mas_fro
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  routing-agent:
    build:
      context: .
      dockerfile: docker/Dockerfile.routing
    depends_on:
      - rabbitmq
      - redis
      - postgres
    environment:
      RABBITMQ_URL: amqp://admin:admin@rabbitmq:5672
      REDIS_URL: redis://redis:6379
      DATABASE_URL: postgresql://postgres:postgres@postgres:5432/mas_fro
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '1.0'
          memory: 2G

  api:
    build:
      context: .
      dockerfile: docker/Dockerfile.api
    ports:
      - "8000:8000"
    depends_on:
      - routing-agent
    environment:
      RABBITMQ_URL: amqp://admin:admin@rabbitmq:5672
      REDIS_URL: redis://redis:6379

volumes:
  rabbitmq_data:
  redis_data:
  postgres_data:
```

### Deliverables for Week 5-6
- [ ] RabbitMQ message broker integrated
- [ ] Docker containers for all services
- [ ] Docker Compose orchestration
- [ ] Horizontal scaling for agents
- [ ] Load balancing configured
- [ ] Performance testing results

---

## 📋 Week 7-8: Monitoring & Observability

### Objective
Add comprehensive monitoring and observability for production readiness.

### Concrete Tasks

#### 4.1 Implement Prometheus Metrics
```python
# src/monitoring/metrics.py
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry
import time

# Create registry
registry = CollectorRegistry()

# Define metrics
route_calculations = Counter(
    'mas_fro_route_calculations_total',
    'Total number of route calculations',
    ['agent_id', 'status'],
    registry=registry
)

route_calculation_duration = Histogram(
    'mas_fro_route_calculation_duration_seconds',
    'Route calculation duration',
    ['agent_id'],
    registry=registry
)

active_agents = Gauge(
    'mas_fro_active_agents',
    'Number of active agents',
    ['agent_type'],
    registry=registry
)

risk_scores = Histogram(
    'mas_fro_risk_scores',
    'Distribution of risk scores',
    ['area'],
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    registry=registry
)

class MetricsCollector:
    """Metrics collection wrapper"""
    
    @staticmethod
    def track_route_calculation(agent_id: str):
        """Decorator to track route calculations"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                start = time.time()
                try:
                    result = func(*args, **kwargs)
                    route_calculations.labels(
                        agent_id=agent_id,
                        status='success'
                    ).inc()
                    return result
                except Exception as e:
                    route_calculations.labels(
                        agent_id=agent_id,
                        status='failure'
                    ).inc()
                    raise
                finally:
                    duration = time.time() - start
                    route_calculation_duration.labels(
                        agent_id=agent_id
                    ).observe(duration)
            return wrapper
        return decorator
```

#### 4.2 Add Grafana Dashboards
```json
// grafana/dashboards/mas-fro.json
{
  "dashboard": {
    "title": "MAS-FRO System Dashboard",
    "panels": [
      {
        "title": "Route Calculations Per Minute",
        "targets": [
          {
            "expr": "rate(mas_fro_route_calculations_total[1m])"
          }
        ],
        "type": "graph"
      },
      {
        "title": "Average Route Calculation Time",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, mas_fro_route_calculation_duration_seconds)"
          }
        ],
        "type": "graph"
      },
      {
        "title": "Active Agents by Type",
        "targets": [
          {
            "expr": "mas_fro_active_agents"
          }
        ],
        "type": "bargauge"
      },
      {
        "title": "Risk Score Distribution",
        "targets": [
          {
            "expr": "mas_fro_risk_scores_bucket"
          }
        ],
        "type": "heatmap"
      }
    ]
  }
}
```

### Deliverables for Week 7-8
- [ ] Prometheus metrics integrated
- [ ] Grafana dashboards configured
- [ ] Alert rules defined
- [ ] Log aggregation with ELK stack
- [ ] Distributed tracing with Jaeger
- [ ] Performance baseline established

---

## 📋 Week 9-10: Production Deployment

### Objective
Deploy to production environment with full CI/CD pipeline.

### Concrete Tasks

#### 5.1 Create CI/CD Pipeline
```yaml
# .github/workflows/deploy.yml
name: Deploy MAS-FRO

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov
      
      - name: Run tests
        run: |
          pytest tests/ --cov=src --cov-report=xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3

  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Build Docker images
        run: |
          docker build -t mas-fro/routing-agent:${{ github.sha }} -f docker/Dockerfile.routing .
          docker build -t mas-fro/api:${{ github.sha }} -f docker/Dockerfile.api .
      
      - name: Push to registry
        run: |
          docker push mas-fro/routing-agent:${{ github.sha }}
          docker push mas-fro/api:${{ github.sha }}

  deploy:
    needs: build
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to Kubernetes
        run: |
          kubectl set image deployment/routing-agent routing-agent=mas-fro/routing-agent:${{ github.sha }}
          kubectl set image deployment/api api=mas-fro/api:${{ github.sha }}
          kubectl rollout status deployment/routing-agent
          kubectl rollout status deployment/api
```

#### 5.2 Kubernetes Production Config
```yaml
# k8s/production/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: routing-agent
  namespace: mas-fro
spec:
  replicas: 5
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 2
      maxUnavailable: 1
  selector:
    matchLabels:
      app: routing-agent
  template:
    metadata:
      labels:
        app: routing-agent
    spec:
      containers:
      - name: routing-agent
        image: mas-fro/routing-agent:latest
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
        env:
        - name: ENVIRONMENT
          value: "production"
        - name: LOG_LEVEL
          value: "INFO"
```

### Deliverables for Week 9-10
- [ ] CI/CD pipeline operational
- [ ] Kubernetes deployment configured
- [ ] Load testing completed
- [ ] Security scanning integrated
- [ ] Documentation complete
- [ ] Production deployment successful

---

## 📊 Success Metrics & KPIs

### Week 1-2 Success Criteria
- Real PAGASA data successfully integrated
- Cache hit rate > 50% for repeated routes
- Zero data validation errors in testing

### Week 3-4 Success Criteria
- API response time < 200ms (p95)
- WebSocket latency < 50ms
- Dashboard loads in < 2 seconds

### Week 5-6 Success Criteria
- Support 1000+ concurrent users
- Message throughput > 5000/sec
- Horizontal scaling demonstrated

### Week 7-8 Success Criteria
- 100% observability coverage
- Alert response time < 5 minutes
- Performance baseline documented

### Week 9-10 Success Criteria
- Zero-downtime deployment achieved
- 99.9% availability maintained
- All security scans passed

---

## 🚀 Quick Start Commands

```bash
# Week 1-2: Set up development environment
pip install -r requirements.txt
docker-compose up -d redis
python src/services/test_data_integration.py

# Week 3-4: Run API and UI
uvicorn src.api.main:app --reload
cd web && npm run dev

# Week 5-6: Start full stack
docker-compose up -d
docker-compose scale routing-agent=3

# Week 7-8: Access monitoring
open http://localhost:3000  # Grafana
open http://localhost:9090  # Prometheus

# Week 9-10: Deploy to production
kubectl apply -f k8s/production/
kubectl get pods -n mas-fro
```

---

## 📚 Additional Resources

### Documentation Templates
- API Documentation: [OpenAPI Spec](docs/openapi.yaml)
- Architecture Decisions: [ADR Template](docs/adr/template.md)
- Runbooks: [Incident Response](docs/runbooks/incident-response.md)

### Training Materials
- [Agent Development Guide](docs/guides/agent-development.md)
- [Performance Tuning Guide](docs/guides/performance-tuning.md)
- [Security Best Practices](docs/guides/security.md)

### External Dependencies
- PAGASA API Documentation: https://api.pagasa.gov.ph/docs
- OpenStreetMap Overpass API: https://wiki.openstreetmap.org/wiki/Overpass_API
- RabbitMQ Best Practices: https://www.rabbitmq.com/best-practices.html

---

*This roadmap provides concrete, actionable steps to transform the MAS-FRO system from prototype to production-ready deployment.*

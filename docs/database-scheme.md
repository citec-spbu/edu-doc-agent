erDiagram
    sessions ||--o{ runs : "включает"

    sessions {
        string id PK
        string status
        datetime created_at
        datetime updated_at
    }

    runs {
        uuid id PK
        string session_id FK
        string status
        datetime last_heartbeat_at
        datetime created_at
    }

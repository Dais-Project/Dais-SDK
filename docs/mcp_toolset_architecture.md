# MCP Toolset 架构图

## 类结构图

```mermaid
classDiagram
    class McpToolset {
        <<abstract>>
        -_client: McpClient | None
        -_tools_cache: list[Tool] | None
        -_connected: bool
        +connect() async
        +disconnect() async
        +refresh_tools() async
        +get_tools() list[ToolFn]
        +is_connected: bool
        #_create_client() McpClient
        #_create_tool_wrapper(mcp_tool) ToolFn
        #_format_tool_result(result) str
    }
    
    class LocalMcpToolset {
        -_name: str
        -_params: LocalServerParams
        +__init__(name, params)
        #_create_client() McpClient
    }
    
    class RemoteMcpToolset {
        -_name: str
        -_params: RemoteServerParams
        +__init__(name, params)
        #_create_client() McpClient
    }
    
    class McpClient {
        <<interface>>
        +connect() async
        +disconnect() async
        +list_tools() async list[Tool]
        +call_tool(name, args) async ToolResult
    }
    
    class LocalMcpClient {
        -_session: ClientSession
        -_params: LocalServerParams
    }
    
    class RemoteMcpClient {
        -_session: ClientSession
        -_params: RemoteServerParams
    }
    
    McpToolset <|-- LocalMcpToolset
    McpToolset <|-- RemoteMcpToolset
    McpToolset o-- McpClient
    McpClient <|.. LocalMcpClient
    McpClient <|.. RemoteMcpClient
    LocalMcpToolset ..> LocalMcpClient : creates
    RemoteMcpToolset ..> RemoteMcpClient : creates
```

## 工作流程图

```mermaid
sequenceDiagram
    participant User
    participant McpToolset
    participant McpClient
    participant McpServer
    participant SDK
    
    User->>McpToolset: __init__(name, params)
    Note over McpToolset: 创建实例，但不连接
    
    User->>McpToolset: await connect()
    McpToolset->>McpClient: _create_client()
    McpToolset->>McpClient: await connect()
    McpClient->>McpServer: 建立连接
    McpServer-->>McpClient: 连接成功
    
    McpToolset->>McpClient: await list_tools()
    McpClient->>McpServer: 请求工具列表
    McpServer-->>McpClient: 返回工具列表
    McpClient-->>McpToolset: list[Tool]
    Note over McpToolset: 缓存工具列表
    Note over McpToolset: 为每个工具创建包装函数
    
    User->>McpToolset: get_tools()
    McpToolset-->>User: list[ToolFn]
    
    User->>SDK: LLM.generate_text(tools=tools)
    Note over SDK: 使用从 MCP 获取的工具
    SDK->>SDK: prepare_tools(tools)
    Note over SDK: 检测到 MCP 工具包装器
    Note over SDK: 使用 MCP 的 inputSchema
    
    SDK->>SDK: 调用 LLM
    Note over SDK: LLM 决定使用工具
    
    SDK->>McpToolset: tool_wrapper(**kwargs)
    McpToolset->>McpClient: await call_tool(name, args)
    McpClient->>McpServer: 执行工具
    McpServer-->>McpClient: ToolResult
    McpClient-->>McpToolset: list[ContentBlock]
    McpToolset->>McpToolset: _format_tool_result()
    McpToolset-->>SDK: str (结果)
    
    SDK-->>User: 响应
    
    User->>McpToolset: await disconnect()
    McpToolset->>McpClient: await disconnect()
    McpClient->>McpServer: 断开连接
    Note over McpToolset: 清理缓存
```

## 工具转换流程

```mermaid
flowchart TD
    A[MCP Tool] --> B{检查工具类型}
    B -->|有 _mcp_tool 属性| C[使用 MCP inputSchema]
    B -->|普通函数| D[解析函数签名]
    B -->|ToolDef| E[解析 execute 签名]
    B -->|RawToolDef| F[直接使用]
    
    C --> G[生成 OpenAI 格式]
    D --> G
    E --> G
    F --> G
    
    G --> H[LLM 工具定义]
    
    style A fill:#e1f5ff
    style C fill:#fff4e1
    style G fill:#e8f5e9
    style H fill:#f3e5f5
```

## 生命周期状态机

```mermaid
stateDiagram-v2
    [*] --> Created: __init__()
    Created --> Connecting: connect()
    Connecting --> Connected: 连接成功
    Connecting --> Created: 连接失败
    Connected --> Connected: refresh_tools()
    Connected --> Disconnecting: disconnect()
    Disconnecting --> Disconnected: 断开成功
    Disconnected --> [*]
    
    note right of Created
        _connected = False
        _client = None
        _tools_cache = None
    end note
    
    note right of Connected
        _connected = True
        _client = McpClient
        _tools_cache = list[Tool]
    end note
```

## 数据流图

```mermaid
flowchart LR
    A[MCP Server] -->|list_tools| B[MCP Client]
    B -->|list of Tool| C[McpToolset]
    C -->|create wrappers| D[Tool Wrappers]
    D -->|get_tool_methods| E[SDK ParamParser]
    E -->|prepare_tools| F[OpenAI Format]
    F -->|send to| G[LLM]
    G -->|tool_call| H[SDK Executor]
    H -->|invoke| D
    D -->|call_tool| B
    B -->|execute| A
    A -->|ToolResult| B
    B -->|ContentBlocks| D
    D -->|format| I[String Result]
    I -->|return to| G
    
    style A fill:#ffebee
    style C fill:#e3f2fd
    style F fill:#f3e5f5
    style G fill:#fff3e0
    style I fill:#e8f5e9
```

## 组件交互图

```mermaid
graph TB
    subgraph "User Code"
        U[User Application]
    end
    
    subgraph "LiteAI SDK"
        L[LLM]
        P[ParamParser]
        PT[prepare_tools]
        E[execute_tool]
    end
    
    subgraph "MCP Toolset Layer"
        LT[LocalMcpToolset]
        RT[RemoteMcpToolset]
        MT[McpToolset Base]
        W[Tool Wrappers]
    end
    
    subgraph "MCP Client Layer"
        LC[LocalMcpClient]
        RC[RemoteMcpClient]
        MC[McpClient Base]
    end
    
    subgraph "External"
        LS[Local MCP Server]
        RS[Remote MCP Server]
    end
    
    U --> L
    L --> P
    P --> LT
    P --> RT
    LT --> MT
    RT --> MT
    MT --> W
    W --> PT
    PT --> L
    L --> E
    E --> W
    
    LT --> LC
    RT --> RC
    LC --> MC
    RC --> MC
    
    LC --> LS
    RC --> RS
    
    style U fill:#e1f5ff
    style L fill:#fff4e1
    style MT fill:#f3e5f5
    style MC fill:#e8f5e9
```

## 错误处理流程

```mermaid
flowchart TD
    A[调用工具] --> B{已连接?}
    B -->|否| C[抛出 RuntimeError]
    B -->|是| D{MCP Client 存在?}
    D -->|否| C
    D -->|是| E[调用 client.call_tool]
    
    E --> F{调用成功?}
    F -->|是| G[格式化结果]
    F -->|否| H{错误类型?}
    
    H -->|连接错误| I[McpSessionNotEstablishedError]
    H -->|工具不存在| J[传播原始错误]
    H -->|参数错误| J
    H -->|执行错误| J
    
    G --> K[返回字符串结果]
    I --> L[用户处理]
    J --> L
    C --> L
    
    style C fill:#ffcdd2
    style I fill:#ffcdd2
    style J fill:#ffcdd2
    style K fill:#c8e6c9
```

## 关键设计模式

### 1. 策略模式 (Strategy Pattern)
- `McpToolset` 定义接口
- `LocalMcpToolset` 和 `RemoteMcpToolset` 提供不同的客户端创建策略

### 2. 工厂方法模式 (Factory Method Pattern)
- `_create_client()` 抽象方法
- 子类决定创建哪种类型的客户端

### 3. 代理模式 (Proxy Pattern)
- 工具包装函数作为 MCP 工具的代理
- 延迟加载和缓存工具列表

### 4. 适配器模式 (Adapter Pattern)
- 将 MCP 工具适配为 SDK 工具系统
- 转换输入/输出格式

```mermaid
graph LR
    A[MCP Tool Format] --> B[McpToolset Adapter]
    B --> C[SDK Tool Format]
    
    D[MCP ToolResult] --> E[_format_tool_result]
    E --> F[String Result]
    
    style B fill:#fff4e1
    style E fill:#fff4e1
# MCP Toolset 架构设计文档

## 1. 概述

本文档描述了 `LocalMcpToolset` 和 `RemoteMcpToolset` 的架构设计，这两个类用于将 MCP (Model Context Protocol) 服务器提供的工具集成到 LiteAI SDK 的工具系统中。

## 2. 核心需求分析

### 2.1 现有系统分析

**Toolset 基类**:
```python
class Toolset:
    def get_tool_methods(self) -> list[ToolFn]:
        """返回所有标记为 @tool 的方法"""
```

**MCP Client**:
```python
class McpClient(ABC):
    async def connect(self): ...
    async def disconnect(self): ...
    async def list_tools(self) -> list[Tool]: ...
    async def call_tool(self, tool_name: str, arguments: dict[str, Any] | None = None) -> ToolResult: ...
```

**MCP Tool 结构**:
- MCP 的 `Tool` 来自 `mcp` 包，包含 `name`、`description` 和 `inputSchema` (JSON Schema 格式)
- MCP 的 `ToolResult` 是 `list[ContentBlock]`，需要转换为字符串

### 2.2 关键挑战

1. **动态工具生成**: MCP 工具是在运行时从服务器获取的，而 Toolset 期望在实例化时就能返回工具方法
2. **异步调用**: MCP 客户端是异步的，但 SDK 的工具系统支持同步和异步两种方式
3. **生命周期管理**: MCP 客户端需要显式连接和断开，需要在 Toolset 中管理这个生命周期
4. **工具签名转换**: MCP 的 JSON Schema 需要转换为 Python 函数签名，以便 SDK 的工具准备系统处理
5. **结果格式转换**: MCP 返回 `list[ContentBlock]`，需要转换为字符串格式

## 3. 架构设计

### 3.1 类层次结构

```
McpToolset (抽象基类)
    ↓
    ├── LocalMcpToolset
    └── RemoteMcpToolset
```

### 3.2 McpToolset 抽象基类设计

```python
class McpToolset(ABC):
    """MCP Toolset 抽象基类 - 独立于 Toolset 类"""
    
    def __init__(self):
        self._client: McpClient | None = None
        self._tools_cache: list[Tool] | None = None
        self._connected: bool = False
    
    @abstractmethod
    def _create_client(self) -> McpClient:
        """创建 MCP 客户端实例"""
    
    async def connect(self) -> None:
        """连接到 MCP 服务器"""
    
    async def disconnect(self) -> None:
        """断开与 MCP 服务器的连接"""
    
    async def refresh_tools(self) -> None:
        """刷新工具列表"""
    
    def get_tools(self) -> list[ToolFn]:
        """返回动态生成的工具函数列表"""
    
    def _create_tool_wrapper(self, mcp_tool: Tool) -> ToolFn:
        """为 MCP 工具创建包装函数"""
```

### 3.3 工具包装机制

MCP 工具需要被包装成符合 SDK 规范的 Python 函数：

```python
def _create_tool_wrapper(self, mcp_tool: Tool) -> ToolFn:
    """
    为 MCP 工具创建包装函数
    
    生成的函数:
    1. 拥有正确的 __name__ 和 __doc__
    2. 使用 **kwargs 接收参数（因为我们无法动态创建具体的参数签名）
    3. 是异步函数，调用 MCP 客户端
    4. 将 MCP 结果转换为字符串
    """
    async def wrapper(**kwargs) -> str:
        if not self._connected or not self._client:
            raise RuntimeError(f"MCP Toolset not connected. Call await {self.__class__.__name__}.connect() first")
        
        result = await self._client.call_tool(mcp_tool.name, kwargs)
        return self._format_tool_result(result)
    
    # 设置函数元数据
    wrapper.__name__ = mcp_tool.name
    wrapper.__doc__ = mcp_tool.description or f"MCP tool: {mcp_tool.name}"
    
    # 标记为工具
    setattr(wrapper, TOOL_FLAG, True)
    
    return wrapper

def _format_tool_result(self, result: ToolResult) -> str:
    """将 MCP ToolResult 转换为字符串"""
    # ToolResult 是 list[ContentBlock]
    # ContentBlock 可能是 TextContent 或 ImageContent 等
    parts = []
    for block in result:
        if hasattr(block, 'text'):
            parts.append(block.text)
        elif hasattr(block, 'data'):
            # 对于其他类型，转换为 JSON
            parts.append(json.dumps(block.model_dump(), ensure_ascii=False))
    return "\n".join(parts)
```

### 3.4 生命周期管理

#### 3.4.1 连接管理

```python
async def connect(self) -> None:
    """连接到 MCP 服务器并缓存工具列表"""
    if self._connected:
        return
    
    self._client = self._create_client()
    await self._client.connect()
    self._connected = True
    
    # 立即获取并缓存工具列表
    await self.refresh_tools()

def get_tools(self) -> list[ToolFn]:
    """获取工具函数列表（必须先调用 connect）"""
    if not self._connected or self._tools_cache is None:
        raise RuntimeError("Not connected to MCP server. Call await connect() first")
    
    return [self._create_tool_wrapper(tool) for tool in self._tools_cache]

async def disconnect(self) -> None:
    """断开连接并清理资源"""
    if self._client and self._connected:
        await self._client.disconnect()
    
    self._client = None
    self._tools_cache = None
    self._connected = False

async def refresh_tools(self) -> None:
    """刷新工具列表（可以在运行时调用）"""
    if not self._connected or not self._client:
        raise RuntimeError("Not connected to MCP server")
    
    mcp_tools = await self._client.list_tools()
    self._tools_cache = mcp_tools
```

#### 3.4.2 上下文管理器支持

```python
async def __aenter__(self):
    await self.connect()
    return self

async def __aexit__(self, exc_type, exc_val, exc_tb):
    await self.disconnect()
```

### 3.5 LocalMcpToolset 实现

```python
class LocalMcpToolset(McpToolset):
    """用于本地 MCP 服务器的 Toolset"""
    
    def __init__(self, name: str, params: LocalServerParams):
        super().__init__()
        self._name = name
        self._params = params
    
    def _create_client(self) -> McpClient:
        return LocalMcpClient(self._name, self._params)
```

### 3.6 RemoteMcpToolset 实现

```python
class RemoteMcpToolset(McpToolset):
    """用于远程 MCP 服务器的 Toolset"""
    
    def __init__(self, name: str, params: RemoteServerParams):
        super().__init__()
        self._name = name
        self._params = params
    
    def _create_client(self) -> McpClient:
        return RemoteMcpClient(self._name, self._params)
```

## 4. 使用示例

### 4.1 基本使用

```python
from liteai_sdk import LLM, LlmRequestParams, UserMessage
from liteai_sdk.tool.toolset import LocalMcpToolset
from liteai_sdk._mcp_client import LocalServerParams

# 创建 MCP toolset
mcp_toolset = LocalMcpToolset(
    name="my-mcp-server",
    params=LocalServerParams(
        command="python",
        args=["-m", "my_mcp_server"]
    )
)

# 连接到 MCP 服务器
await mcp_toolset.connect()

try:
    # 使用 toolset
    llm = LLM(provider=LlmProviders.OPENAI, api_key="...")
    
    # 获取工具列表
    tools = mcp_toolset.get_tools()
    
    response = await llm.generate_text(
        LlmRequestParams(
            model="gpt-4",
            messages=[UserMessage(content="Use the MCP tools")],
            tools=tools  # 直接传递工具列表
        )
    )
finally:
    # 断开连接
    await mcp_toolset.disconnect()
```

### 4.2 使用上下文管理器

```python
async with LocalMcpToolset(name="...", params=...) as mcp_toolset:
    tools = mcp_toolset.get_tools()
    response = await llm.generate_text(
        LlmRequestParams(
            model="gpt-4",
            messages=[UserMessage(content="...")],
            tools=tools
        )
    )
```

### 4.3 远程 MCP 服务器

```python
from liteai_sdk._mcp_client import RemoteServerParams

mcp_toolset = RemoteMcpToolset(
    name="remote-server",
    params=RemoteServerParams(
        url="https://mcp-server.example.com",
        bearer_token="your-token"
    )
)

await mcp_toolset.connect()
tools = mcp_toolset.get_tools()
# ... 使用 tools
await mcp_toolset.disconnect()
```

## 5. 关键设计决策

### 5.1 为什么使用懒加载工具列表？

**决策**: 工具列表在 `connect()` 时获取并缓存，而不是在 `__init__` 时。

**原因**:
1. MCP 客户端需要异步连接，而 `__init__` 是同步的
2. 允许用户控制何时建立连接（节省资源）
3. 支持工具列表的动态刷新

### 5.2 为什么使用 **kwargs 而不是动态生成函数签名？

**决策**: 包装函数使用 `**kwargs` 接收参数。

**原因**:
1. Python 动态生成具有特定签名的函数非常复杂
2. SDK 的 `prepare_tools` 会解析函数签名，但主要用于生成 JSON Schema
3. MCP 已经提供了 JSON Schema (`inputSchema`)，可以直接使用
4. 简化实现，减少错误

### 5.3 为什么不继承 Toolset 类？

**决策**: `McpToolset` 不继承 `Toolset` 类，而是独立实现。

**原因**:
1. MCP 工具集的生命周期管理（连接/断开）与普通 Toolset 不同
2. MCP 工具是动态获取的，不适合使用 `@tool` 装饰器模式
3. 使用 `get_tools()` 而不是 `get_tool_methods()` 更符合 MCP 的语义
4. 保持 API 的独立性和灵活性

### 5.4 如何处理 MCP 的 inputSchema？

**方案**: 由于包装函数使用 `**kwargs`，SDK 的 `prepare_tools` 会将其处理为接受任意参数。我们需要修改 `prepare_tools` 来特殊处理 MCP 工具。

**建议实现**:
```python
# 在 prepare.py 中添加
def generate_tool_definition_from_mcp_tool(mcp_tool: Tool) -> dict[str, Any]:
    """Convert MCP Tool to OpenAI tools format"""
    return {
        "type": "function",
        "function": {
            "name": mcp_tool.name,
            "description": mcp_tool.description or f"MCP tool: {mcp_tool.name}",
            "parameters": mcp_tool.inputSchema  # 直接使用 MCP 的 JSON Schema
        }
    }
```

然后在包装函数上附加原始的 MCP Tool 对象：
```python
def _create_tool_wrapper(self, mcp_tool: Tool) -> ToolFn:
    async def wrapper(**kwargs) -> str:
        # ... 实现
    
    wrapper.__name__ = mcp_tool.name
    wrapper.__doc__ = mcp_tool.description or f"MCP tool: {mcp_tool.name}"
    setattr(wrapper, TOOL_FLAG, True)
    setattr(wrapper, "_mcp_tool", mcp_tool)  # 附加 MCP Tool 对象
    
    return wrapper
```

然后在 `prepare_tools` 中检测这个属性：
```python
def prepare_tools(tools: Sequence[ToolLike]) -> list[dict]:
    tool_defs = []
    for tool in tools:
        # 检查是否是 MCP 工具包装器
        if callable(tool) and hasattr(tool, "_mcp_tool"):
            mcp_tool = getattr(tool, "_mcp_tool")
            tool_defs.append(generate_tool_definition_from_mcp_tool(mcp_tool))
        elif callable(tool):
            tool_defs.append(generate_tool_definition_from_callable(tool))
        elif isinstance(tool, ToolDef):
            tool_defs.append(generate_tool_definition_from_tool_def(tool))
        else:
            tool_defs.append(generate_tool_definition_from_raw_tool_def(tool))
    return tool_defs
```

### 5.4 错误处理策略

1. **未连接错误**: 如果调用工具时未连接，抛出清晰的错误信息
2. **MCP 错误传播**: MCP 客户端的错误应该透明地传播给用户
3. **断开连接容错**: `disconnect()` 应该处理部分失败的情况

## 6. 测试策略

### 6.1 单元测试

1. 测试工具包装函数的生成
2. 测试生命周期管理（连接/断开）
3. 测试工具列表缓存和刷新
4. 测试结果格式转换

### 6.2 集成测试

1. 使用模拟 MCP 服务器测试完整流程
2. 测试与 SDK 工具系统的集成
3. 测试错误情况（连接失败、工具调用失败等）

## 7. 未来扩展

### 7.1 工具过滤

允许用户选择性地包含某些工具：

```python
class McpToolset:
    def __init__(self, ..., tool_filter: Callable[[Tool], bool] | None = None):
        self._tool_filter = tool_filter
    
    async def refresh_tools(self) -> None:
        mcp_tools = await self._client.list_tools()
        if self._tool_filter:
            mcp_tools = [t for t in mcp_tools if self._tool_filter(t)]
        self._tools_cache = mcp_tools
```

### 7.2 工具重命名

允许用户重命名工具以避免冲突：

```python
class McpToolset:
    def __init__(self, ..., tool_prefix: str = ""):
        self._tool_prefix = tool_prefix
    
    def _create_tool_wrapper(self, mcp_tool: Tool) -> ToolFn:
        # ...
        wrapper.__name__ = f"{self._tool_prefix}{mcp_tool.name}"
```

### 7.3 自动重连

在连接断开时自动重连：

```python
class McpToolset:
    def __init__(self, ..., auto_reconnect: bool = False):
        self._auto_reconnect = auto_reconnect
    
    async def _call_with_retry(self, tool_name: str, arguments: dict):
        try:
            return await self._client.call_tool(tool_name, arguments)
        except Exception as e:
            if self._auto_reconnect:
                await self.connect()
                return await self._client.call_tool(tool_name, arguments)
            raise
```

## 8. 实现清单

- [ ] 创建 `mcp_toolset.py` 文件
- [ ] 实现 `McpToolset` 抽象基类
- [ ] 实现 `LocalMcpToolset`
- [ ] 实现 `RemoteMcpToolset`
- [ ] 修改 `prepare.py` 以支持 MCP 工具
- [ ] 更新 `__init__.py` 导出新类
- [ ] 编写单元测试
- [ ] 编写使用示例
- [ ] 更新文档

## 9. API 参考

### McpToolset

```python
class McpToolset(ABC):
    async def connect(self) -> None:
        """连接到 MCP 服务器"""
    
    async def disconnect(self) -> None:
        """断开连接"""
    
    async def refresh_tools(self) -> None:
        """刷新工具列表"""
    
    def get_tools(self) -> list[ToolFn]:
        """获取工具函数列表（必须先调用 connect）"""
    
    @property
    def is_connected(self) -> bool:
        """检查是否已连接"""
```

### LocalMcpToolset

```python
class LocalMcpToolset(McpToolset):
    def __init__(self, name: str, params: LocalServerParams):
        """
        Args:
            name: MCP 服务器名称
            params: 本地服务器参数
        """
```

### RemoteMcpToolset

```python
class RemoteMcpToolset(McpToolset):
    def __init__(self, name: str, params: RemoteServerParams):
        """
        Args:
            name: MCP 服务器名称
            params: 远程服务器参数
        """
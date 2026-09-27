// Control/read deadlines must not shorten creative generation or streaming calls.
export async function withRequestTimeout<T>(operation: (signal: AbortSignal) => Promise<T>, milliseconds: number): Promise<T> {
  const controller = new AbortController();
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    return await Promise.race([
      operation(controller.signal),
      new Promise<never>((_, reject) => {
        timer = setTimeout(() => {
          reject(new Error("读取或控制请求超时；操作可能已受理，请刷新状态后核对"));
          controller.abort();
        }, milliseconds);
      }),
    ]);
  } finally { clearTimeout(timer); }
}

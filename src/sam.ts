import child_process from 'child_process'

const sam = child_process.spawn('python', ['sam.py'], {
  stdio: ['pipe', 'pipe', 'inherit'],
  shell: true
})

let lock = false
let state: 'start' | 'file' | 'inited' = 'start'
let stateTransferred: () => void = () => {}
let query: (ret: [number, number, number, number]) => void = () => {}

sam.stdout.on('data', (chunk: string) => {
  const cmd = chunk.toString().trim()
  if (cmd.startsWith('-file')) {
    state = 'file'
    stateTransferred()
  } else if (cmd.startsWith('-loaded')) {
    state = 'inited'
    stateTransferred()
  } else if (cmd.startsWith('-result')) {
    state = 'inited'
    query(JSON.parse(cmd.substring(7)))
  }
})

async function wait_state() {
  return new Promise<void>(resolve => {
    stateTransferred = resolve
  })
}

async function break_state() {
  return new Promise<void>(resolve => {
    stateTransferred = resolve
    sam.stdin.write('-q\n')
  })
}

async function set_file(p: string) {
  return new Promise<void>(resolve => {
    stateTransferred = resolve
    sam.stdin.write(`${p}\n`)
  })
}

async function query_res(p: [number, number][]) {
  return new Promise<[number, number, number, number]>(resolve => {
    query = resolve
    sam.stdin.write(`${JSON.stringify(p)}\n`)
  })
}

export async function load_file(p: string) {
  if (lock) {
    return
  }
  lock = true
  if (state == 'inited') {
    await break_state()
  }
  if (state == 'start') {
    await wait_state()
  }
  await set_file(p)
  lock = false
}

export async function perform_predict(pts: [number, number][]) {
  if (lock) {
    return
  }
  if (state != 'inited') {
    return
  }
  lock = true
  const res = await query_res(pts)
  lock = false
  return res
}

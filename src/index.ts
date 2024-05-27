import express, { json } from 'express'
import { load_file, perform_predict } from './sam'
import cors from 'cors'
import bodyParser from 'body-parser'
import fs from 'fs/promises'

const app = express()

app.use(json())
app.use(
  bodyParser.raw({
    type: 'image/png',
    limit: '100mb'
  })
)
app.use(
  cors({
    origin: '*'
  })
)

app.post('/file', async (req, res) => {
  await fs.writeFile('image.png', req.body)
  await load_file('image.png')
  res.send({})
})

app.post('/query', async (req, res) => {
  const pts = req.body as { pts: [number, number][] }
  console.log(req.body)
  res.send(await perform_predict(pts.pts))
})

app.listen(13127)

// async function main() {
//   await load_file('images/truck.jpg')
//   console.log(
//     await perform_predict([
//       [500, 375],
//       [350, 375]
//     ])
//   )
// }

// main()

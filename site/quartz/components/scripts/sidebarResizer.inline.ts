// Sidebar resizer: drag the edge between a sidebar and the article to set its
// width. Widths persist in localStorage and survive SPA navigation.

const MIN_WIDTH = 200
const MAX_WIDTH = 500
const DEFAULT_WIDTH = 320

const LEFT_KEY = "sidebar-left-width"
const RIGHT_KEY = "sidebar-right-width"

const clamp = (n: number) => Math.max(MIN_WIDTH, Math.min(MAX_WIDTH, n))

const applyWidth = (side: "left" | "right", px: number) => {
  document.documentElement.style.setProperty(
    `--sidebar-${side}-width`,
    `${px}px`,
  )
}

const loadSavedWidths = () => {
  const lsLeft = parseInt(localStorage.getItem(LEFT_KEY) ?? "")
  const lsRight = parseInt(localStorage.getItem(RIGHT_KEY) ?? "")
  if (!Number.isNaN(lsLeft)) applyWidth("left", clamp(lsLeft))
  if (!Number.isNaN(lsRight)) applyWidth("right", clamp(lsRight))
}

// Apply saved widths as early as possible, before the first paint of the
// next page, so resizes don't visually pop on every navigation.
loadSavedWidths()

const ensureResizer = (side: "left" | "right") => {
  const sidebar = document.querySelector(
    `.sidebar.${side}`,
  ) as HTMLElement | null
  if (!sidebar) return

  // Skip if already injected for this sidebar.
  if (sidebar.querySelector(`.sidebar-resizer[data-side="${side}"]`)) return

  const handle = document.createElement("div")
  handle.className = "sidebar-resizer"
  handle.dataset.side = side
  handle.setAttribute("aria-hidden", "true")
  sidebar.appendChild(handle)

  let dragging = false

  const onMouseDown = (e: MouseEvent) => {
    dragging = true
    e.preventDefault()
    document.body.classList.add("sidebar-resizing")
  }

  const onMouseMove = (e: MouseEvent) => {
    if (!dragging) return
    // For the left sidebar, width = distance from viewport-left to mouse.
    // For the right sidebar, width = distance from mouse to viewport-right.
    const px =
      side === "left" ? e.clientX : window.innerWidth - e.clientX
    const next = clamp(px)
    applyWidth(side, next)
  }

  const onMouseUp = () => {
    if (!dragging) return
    dragging = false
    document.body.classList.remove("sidebar-resizing")
    const cssValue = getComputedStyle(document.documentElement).getPropertyValue(
      `--sidebar-${side}-width`,
    )
    const px = parseInt(cssValue)
    if (!Number.isNaN(px)) {
      localStorage.setItem(side === "left" ? LEFT_KEY : RIGHT_KEY, `${px}`)
    }
  }

  handle.addEventListener("mousedown", onMouseDown)
  document.addEventListener("mousemove", onMouseMove)
  document.addEventListener("mouseup", onMouseUp)

  // Double-click resets that side to the default width.
  const onDoubleClick = () => {
    applyWidth(side, DEFAULT_WIDTH)
    localStorage.removeItem(side === "left" ? LEFT_KEY : RIGHT_KEY)
  }
  handle.addEventListener("dblclick", onDoubleClick)

  window.addCleanup(() => {
    handle.removeEventListener("mousedown", onMouseDown)
    handle.removeEventListener("dblclick", onDoubleClick)
    document.removeEventListener("mousemove", onMouseMove)
    document.removeEventListener("mouseup", onMouseUp)
  })
}

document.addEventListener("nav", () => {
  loadSavedWidths()
  ensureResizer("left")
  ensureResizer("right")
})

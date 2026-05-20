let leftHidden = false
let rightHidden = false

document.addEventListener("nav", () => {
  const apply = (side: "left" | "right") => {
    const hidden = side === "left" ? leftHidden : rightHidden
    document.documentElement.setAttribute(
      `${side}-sidebar`,
      hidden ? "hidden" : "shown",
    )
  }

  const toggle = (side: "left" | "right") => {
    if (side === "left") {
      leftHidden = !leftHidden
    } else {
      rightHidden = !rightHidden
    }
    apply(side)
  }

  for (const btn of document.getElementsByClassName("readermode")) {
    const side = ((btn as HTMLElement).dataset.side ?? "left") as "left" | "right"
    const handler = () => toggle(side)
    btn.addEventListener("click", handler)
    window.addCleanup(() => btn.removeEventListener("click", handler))
  }

  apply("left")
  apply("right")
})

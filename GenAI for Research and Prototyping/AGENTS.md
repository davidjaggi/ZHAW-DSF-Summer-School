# GenAI for Research and Prototyping — Workshop Vault

ZHAW DSF Summer School. Obsidian vault for the "Switzerland's Future Energy Mix" case study.

## Structure

```
Notes/       # Your active case-study work
Resources/   # Reference material: slides, brief, templates, instructor's solution
```

## The CODE process

Capture → Organize → Distill → Express (Tiago Forte's framework). This is a **process
you apply to notes, not a folder structure** — don't create `Capture/`, `Organize/`,
`Distill/`, `Express/` subfolders. Notes live in `Notes/<project>/` and move through the
four stages in place.

- **Capture** — pull raw material from primary sources, always cited
- **Organize** — sort captured material against the report structure
- **Distill** — rewrite into linked, atomic notes (one idea per note)
- **Express** — assemble the final report/dashboard from distilled notes

## Conventions

- Markdown throughout; use Obsidian `[[wikilinks]]` for internal references
- Note templates (Concept, Entity, Source) live in `Resources/Templates/`
- Cite sources — every claim should trace back to where it came from
- Reference material in `Resources/` is read-only; don't edit the instructor's
  worked-example solution in `Resources/solution/`

## The case study

See `Resources/Course Materials/0. Casestudy.md` for the full brief: recommend an energy
mix for Switzerland covering cost, environmental impact, reliability, scalability,
implementation time, land use, public acceptance, and energy security.

## Running the solution dashboard

If asked to run or extend the worked-example dashboard in `Resources/solution/`, see
`Resources/solution/CLAUDE.md` for the Python environment and Streamlit setup notes
(local conda env path, headless flag, first-run credentials workaround).

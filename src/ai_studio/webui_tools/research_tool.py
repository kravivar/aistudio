import json
from ai_studio.notebook.manager import notebook_manager
from ai_studio.notebook.synthesis import synthesis_engine

class Tools:
    """
    Open WebUI Agent Tool: Open Notebook Research & Synthesis Engine.
    Allows LLMs in Open WebUI to manage notes, search research, summarize documents,
    generate study flashcards, and write multi-speaker podcasts.
    """
    def __init__(self):
        pass

    def save_research_note(self, title: str, content: str, tags: str = "") -> str:
        """
        Saves a research note to the local notebook database.
        :param title: Title of the research note
        :param content: Main content of the research note
        :param tags: Comma-separated tags
        """
        tag_list = [t.strip() for t in tags.split(",") if t.strip()]
        res = notebook_manager.create_note(title=title, content=content, tags=tag_list)
        return f"✅ Saved note '{title}' with ID {res['id']}"

    def search_research_notes(self, query: str) -> str:
        """
        Searches existing research notes in the notebook database.
        :param query: Keyword or topic to search for
        """
        results = notebook_manager.search_notes(query)
        if not results:
            return f"No notes found matching query: '{query}'"
        
        output = [f"Found {len(results)} matching research note(s):"]
        for n in results:
            output.append(f"📌 [{n['title']}] (ID: {n['id']})\n{n['content'][:300]}...\nTags: {', '.join(n['tags'])}\n")
        return "\n".join(output)

    def synthesize_document(self, action: str, content: str) -> str:
        """
        Performs deep research synthesis on document text using Open Notebook engines.
        :param action: Synthesis task - 'summarize', 'expand', 'questions', or 'podcast'
        :param content: Document text to process
        """
        action_lower = action.lower()
        if "summary" in action_lower or "summarize" in action_lower:
            res = synthesis_engine.summarize(content)
        elif "expand" in action_lower or "elaborate" in action_lower:
            res = synthesis_engine.expand(content)
        elif "question" in action_lower or "flashcard" in action_lower:
            res = synthesis_engine.generate_questions(content)
        elif "podcast" in action_lower:
            res = synthesis_engine.generate_podcast_script(content)
        else:
            return f"Unknown synthesis action '{action}'. Valid actions are: summarize, expand, questions, podcast."

        choices = res.get("choices", [])
        if choices:
            return choices[0].get("message", {}).get("content", "No output generated.")
        return "No response received from synthesis engine."

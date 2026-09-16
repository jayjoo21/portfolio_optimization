<script setup lang="ts">
import { RouterLink, RouterView } from 'vue-router'
</script>

<template>
  <div class="app-shell">
    <aside class="sidebar">
      <div class="sidebar-main">
        <!-- BRAND -->
        <RouterLink to="/" class="brand">
          <strong> Portfolio </strong>

          <span> Optimization </span>
        </RouterLink>

        <!-- NAVIGATION -->
        <nav class="navigation">
          <p class="nav-section">WORKSPACE</p>

          <RouterLink to="/"> Overview </RouterLink>

          <RouterLink to="/replay"> Decision Replay </RouterLink>

          <RouterLink to="/backtest"> Backtest </RouterLink>

          <RouterLink to="/execution"> Execution </RouterLink>
        </nav>
      </div>

      <!-- FOOTER -->
      <div class="sidebar-footer">
        <span class="prototype-tag"> PROTOTYPE · MOCK DATA </span>

        <div class="model-status">
          <span class="status-dot"></span>

          MODEL
        </div>

        <strong> Deep RL · EIIE </strong>

        <small> Historical simulation </small>
      </div>
    </aside>

    <section class="content">
      <RouterView v-slot="{ Component }">
        <Transition name="page" mode="out-in">
          <component :is="Component" />
        </Transition>
      </RouterView>
    </section>
  </div>
</template>

<style scoped>
.app-shell {
  display: flex;
  min-height: 100vh;
}

/* -------------------------
   SIDEBAR
------------------------- */

.sidebar {
  position: sticky;
  top: 0;

  display: flex;

  width: 218px;
  height: 100vh;

  flex-shrink: 0;
  flex-direction: column;
  justify-content: space-between;

  padding: 30px 20px 24px;

  border-right: 1px solid var(--border);

  background: rgba(255, 255, 255, 0.94);

  backdrop-filter: blur(18px);
}

.sidebar-main {
  width: 100%;
}

/* -------------------------
   BRAND
------------------------- */

.brand {
  display: flex;

  flex-direction: column;

  gap: 1px;

  width: fit-content;

  padding: 2px 8px;

  transition: opacity 150ms ease;
}

.brand:hover {
  opacity: 0.7;
}

.brand strong {
  color: var(--text);

  font-size: 16px;
  font-weight: 750;

  letter-spacing: -0.035em;
}

.brand span {
  color: var(--text-muted);

  font-size: 10px;

  letter-spacing: 0.01em;
}

/* -------------------------
   NAVIGATION
------------------------- */

.navigation {
  display: flex;

  flex-direction: column;

  gap: 2px;

  margin-top: 50px;
}

.nav-section {
  margin: 0 0 10px 9px;

  color: #adb3c0;

  font-size: 9px;
  font-weight: 700;

  letter-spacing: 0.12em;
}

.navigation a {
  position: relative;

  display: flex;

  align-items: center;

  min-height: 39px;

  padding: 0 10px;

  border-radius: 7px;

  color: #697386;

  font-size: 12px;
  font-weight: 500;

  transition:
    color 150ms ease,
    background 150ms ease;
}

.navigation a:hover {
  color: var(--text);

  background: #f7f8fb;
}

.navigation a.router-link-active {
  color: var(--primary);

  background: var(--primary-soft);

  font-weight: 650;
}

.navigation a.router-link-active::before {
  position: absolute;

  top: 9px;
  bottom: 9px;
  left: 0;

  width: 2px;

  border-radius: 999px;

  background: var(--primary);

  content: '';
}

/* -------------------------
   FOOTER
------------------------- */

.sidebar-footer {
  display: flex;

  flex-direction: column;

  gap: 4px;

  padding: 18px 8px 4px;

  border-top: 1px solid var(--border);
}

.prototype-tag {
  width: fit-content;

  margin-bottom: 12px;
  padding: 5px 8px;

  border: 1px solid var(--primary-border);
  border-radius: 999px;

  background: var(--primary-soft);

  color: var(--primary);

  font-size: 8px;
  font-weight: 750;

  letter-spacing: 0.08em;
}

.model-status {
  display: flex;

  align-items: center;

  gap: 6px;

  color: var(--text-muted);

  font-size: 9px;
  font-weight: 700;

  letter-spacing: 0.1em;
}

.status-dot {
  width: 6px;
  height: 6px;

  border-radius: 50%;

  background: var(--success);

  box-shadow: 0 0 0 3px rgba(47, 158, 98, 0.1);
}

.sidebar-footer strong {
  margin-top: 3px;

  font-size: 12px;
}

.sidebar-footer small {
  color: var(--text-muted);

  font-size: 10px;
}

/* -------------------------
   CONTENT
------------------------- */

.content {
  min-width: 0;

  flex: 1;
}

/* -------------------------
   RESPONSIVE
------------------------- */

@media (max-width: 800px) {
  .sidebar {
    width: 164px;

    padding: 24px 14px;
  }
}
</style>

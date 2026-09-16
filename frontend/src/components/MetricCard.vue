<script setup lang="ts">
defineProps<{
  label: string
  value: string
}>()
</script>

<template>
  <div class="metric-card">
    <p>{{ label }}</p>
    <strong>{{ value }}</strong>
  </div>
</template>

<style scoped>
.metric-card {
  padding: 24px;
  border: 1px solid #e4e4e7;
  border-radius: 16px;
  background: #ffffff;
}

.metric-card p {
  margin: 0 0 12px;
  color: #71717a;
  font-size: 13px;
}

.metric-card strong {
  font-size: 20px;
}
</style>
